from rllab.misc import tensor_utils
import numpy as np
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
from sandbox.adam.gpar.sampler.base import BaseGpuSampler
from sandbox.adam.gpar.sampler.prog_bar import ProgBarCounter
import gtimer as gt
import psutil
from rllab.misc import ext
import copy


class PairwiseGpuMultiSampler(BaseGpuSampler):

    def __init__(
        self, algo, n_parallel,
        n_simulators=1,
        set_cpu_affinity=True,
        cpu_assignments=None
    ):
        """
        :type algo: GpuBatchPolopt
        """
        self.algo = algo
        self.batch_size = self.algo.batch_size
        self.n_parallel = n_parallel
        self.n_simulators = n_simulators
        self.set_cpu_affinity = set_cpu_affinity
        self.cpu_assignments = cpu_assignments
        self.initialized = False

    def initialize_par_objs(self):
        """ Build all parallel objects """
        n_par = self.n_parallel
        n = n_par * self.n_simulators
        obs_dim = int(self.algo.env.observation_space.flat_dim)
        act_dim = int(self.algo.env.action_space.flat_dim)

        shutdown = mp.RawValue(c_bool, False)
        barrier = mp.Barrier(n_par + 1)
        # NOTE: using typecode 'f' for float32.
        shareds = struct(
            shutdown=mp.RawValue(c_bool, False),
            obs=[np.ctypeslib.as_array(
                mp.RawArray('f', n * obs_dim)).reshape(n, obs_dim)
                for _ in range(2)],
            act=[np.ctypeslib.as_array(
                mp.RawArray('f', n * act_dim)).reshape(n, act_dim)
                for _ in range(2)],
            rew=[np.ctypeslib.as_array(mp.RawArray('f', n)) for _ in range(2)],
            done=[mp.RawArray(c_bool, n) for _ in range(2)],
            continue_sampling=mp.RawValue(c_bool, True),
        )
        semaphores = struct(
            step_blockers=([mp.Semaphore(0) for _ in range(n_par)],
                           [mp.Semaphore(0) for _ in range(n_par)]),
            act_waiters=([mp.Semaphore(0) for _ in range(n_par)],
                         [mp.Semaphore(0) for _ in range(n_par)]),
        )
        self.par_objs = struct(shutdown=shutdown, barrier=barrier,
            shareds=shareds)  # workers inherit

        workers = []
        for rank in range(n_par):
            rank_semas = struct(
                step_blocker=(semaphores.step_blockers[0][rank],
                               semaphores.step_blockers[1][rank]),
                act_waiter=(semaphores.act_waiters[0][rank],
                             semaphores.act_waiters[1][rank]),
            )
            workers.append(mp.Process(target=self.sampling_worker,
                                      args=(rank, rank_semas)))

        self.par_objs.semaphores = semaphores  # workers don't inherit
        self.par_objs.workers = workers
        if self.set_cpu_affinity:
            self.par_objs.master_process = psutil.Process()
            self.all_cpus = list(range(psutil.cpu_count()))
        self.initialized = True

    def initialize_worker(self, rank):
        if self.set_cpu_affinity:
            self.set_worker_cpu_affinity(rank)
        seed = ext.get_seed()
        if seed is None:
            # NOTE: not sure if this is a good source for seed?
            seed = int(1e6 * np.random.rand())
        ext.set_seed(seed + rank)
        self.envs = ([copy.deepcopy(self.algo.env) for _ in range(self.n_simulators)],
                     [copy.deepcopy(self.algo.env) for _ in range(self.n_simulators)])
        start_idx = rank * self.n_simulators
        self.idx = [start_idx + s for s in range(self.n_simulators)]

    @gt.wrap
    def obtain_samples(self, itr):
        """ Only master (GPU) thread executes this method """
        if self.set_cpu_affinity:
            self.par_objs.master_process.cpu_affinity([0])
        self.par_objs.barrier.wait()  # signal workers to re-enter
        shareds = self.par_objs.shareds
        act_waiters = self.par_objs.semaphores.act_waiters
        step_blockers = self.par_objs.semaphores.step_blockers

        shareds.continue_sampling.value = True  # (set before workers check)
        continue_sampling = True

        paths = []
        cum_length_complete_paths = 0
        range_sims = range(self.n_parallel * self.n_simulators)
        sims_act = [[[] for _ in range_sims] for _ in range(2)]
        sims_obs = [[[] for _ in range_sims] for _ in range(2)]
        sims_rew = [[[] for _ in range_sims] for _ in range(2)]
        sims_agent_info = [[[] for _ in range_sims] for _ in range(2)]

        all_obs = [None] * 2
        all_act = [None] * 2
        all_agent_info = [None] * 2
        all_rew = [None] * 2
        all_done = [None] * 2
        next_obs = [None] * 2
        next_agent_info = [None] * 2

        pbar = ProgBarCounter(self.batch_size)

        [b.acquire() for b in step_blockers[0]]

        # Start-up: do not record yet.
        all_obs[0] = shareds.obs[0].copy()
        o = self.algo.env.observation_space.unflatten_n(all_obs[0])
        act, all_agent_info[0] = self.algo.policy.get_actions(o)  # could instead send shareds.current_obs
        all_act[0] = self.algo.env.action_space.flatten_n(act)
        shareds.act[0][:] = all_act[0]  # copy just for building the paths.

        [w.release() for w in act_waiters[0]]
        [b.acquire() for b in step_blockers[1]]

        all_obs[1] = shareds.obs[1].copy()
        o = self.algo.env.observation_space.unflatten_n(all_obs[1])
        act, all_agent_info[1] = self.algo.policy.get_actions(o)
        all_act[1] = self.algo.env.action_space.flatten_n(act)
        shareds.act[1][:] = all_act[1]

        [waiter.release() for waiter in act_waiters[1]]  # gates.act[1].open()

        gt.stamp('startup')
        # loop = gt.timed_loop('samp', save_itrs=False)
        i = -1
        j = 1
        while continue_sampling:
            # next(loop)
            i += 1
            j = j ^ 1  # xor -- toggles
            # time.sleep(0.01)

            [b.acquire() for b in step_blockers[j]]

            all_rew[j] = shareds.rew[j].copy()
            next_obs[j] = shareds.obs[j].copy()
            all_done[j] = shareds.done[j][:]

            # gt.stamp('copy')
            o = self.algo.env.observation_space.unflatten_n(next_obs[j])
            act, next_agent_info[j] = self.algo.policy.get_actions(o)
            shareds.act[j][:] = self.algo.env.action_space.flatten_n(act)
            # gt.stamp('get_act')

            [w.release() for w in act_waiters[j]]

            # Move current data into persistent variables.
            for idx in range_sims:
                sims_act[j][idx].append(all_act[j][idx, :])
                sims_rew[j][idx].append(all_rew[j][idx])
                sims_obs[j][idx].append(all_obs[j][idx, :])
                sims_agent_info[j][idx].append({k: v[idx] for k, v in all_agent_info[j].items()})
                if all_done[j][idx]:
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs[j][idx]),
                        actions=tensor_utils.stack_tensor_list(sims_act[j][idx]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew[j][idx]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info[j][idx]),
                        env_infos={},  # (have yet to see an env provide this)
                    ))
                    cum_length_complete_paths += len(sims_rew[j][idx])
                    sims_act[j][idx] = []
                    sims_obs[j][idx] = []
                    sims_rew[j][idx] = []
                    sims_agent_info[j][idx] = []
            if any(all_done[j]):
                continue_sampling = (cum_length_complete_paths < self.batch_size)
                pbar.update(cum_length_complete_paths)
            all_obs[j] = next_obs[j]
            all_act[j] = shareds.act[j][:]
            all_agent_info[j] = next_agent_info[j]

        # loop.exit()
        gt.stamp('samp')
        print("Master exited sampling loop: ", itr, "at count: ", i)

        # Simulators do 2 more steps, since the act_gates have already been
        # opened.  Wait for them to do the 2nd step (the same j) and get stopped
        # at the next act_gate.
        [b.acquire() for b in step_blockers[j]]
        # Now the simulators are all waiting at the action of (j ^ 1)
        shareds.continue_sampling.value = False
        j = j ^ 1
        [w.release() for w in act_waiters[j]]
        # Now the simulators check the while condition and exit.
        [b.acquire() for b in step_blockers[j]]  # (close the gate)
        pbar.stop()
        if self.set_cpu_affinity:
            self.par_objs.master_process.cpu_affinity(self.all_cpus)

        return paths

    @gt.wrap
    def obtain_samples_worker(self, rank, semaphores):
        """ Workers execute synchronously with master doing obtain_samples() """

        shareds = self.par_objs.shareds
        step_blocker = semaphores.step_blocker  # (a pair)
        act_waiter = semaphores.act_waiter  # (a pair)
        envs = self.envs  # (a pair of lists)
        idx = self.idx

        # Start-up: provide first observations.
        for s, env in zip(idx, envs[0]):
            shareds.obs[0][s, :] = env.observation_space.flatten(env.reset())
        step_blocker[0].release()
        for s, env in zip(idx, envs[1]):
            shareds.obs[1][s, :] = env.observation_space.flatten(env.reset())
        step_blocker[1].release()
        gt.stamp('first_obs')
        act_waiter[0].acquire()
        gt.stamp('bar_first_act')

        loop = gt.timed_loop('samp', save_itrs=False)
        # i = -1
        j = 0
        while shareds.continue_sampling.value:
            next(loop)
            # Synchronization diagnostic / test:
            # i += 1
            # print(rank, i)
            # if rank == 0:
            #     time.sleep(0.01)

            for s, env in zip(idx, envs[j]):
                a = env.action_space.unflatten(shareds.act[j][s, :])
                o, r, d, env_info = env.step(a)

                # gt.stamp('step')
                # TODO: Later, might want to have the simulator skip an iteration to reset.
                if d:
                    o = env.reset()
                    # gt.stamp('reset')

                # Share all the current data.
                shareds.obs[j][s, :] = env.observation_space.flatten(o)
                shareds.rew[j][s] = r
                shareds.done[j][s] = d
                # (have yet to see an environment provide env_info)

            step_blocker[j].release()
            j = j ^ 1  # xor -- toggles
            gt.stamp('all_but_waiter')
            act_waiter[j].acquire()
            gt.stamp('just_waiter')

            # gt.stamp('bar_act')
        loop.exit()
        # gt.stamp('samp', qp=False)
        # Every rank should print the same iteration count, 2 greater than the
        # iteration count of the master.
        # print("Rank: ", rank, "exited sampling loop: ", itr, "at count: ", i)
