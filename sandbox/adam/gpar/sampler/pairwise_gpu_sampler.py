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
import time
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete


def get_space_properties(space_obj):
    if isinstance(space_obj, Box):
        shape = space_obj.shape
        size = int(np.prod(shape))
        typecode = 'f'
    elif isinstance(space_obj, Discrete):
        shape = tuple()
        size = 1
        typecode = 'i'
    else:
        raise TypeError("Unsupported Space: {}".format(type(space_obj)))

    return shape, size, typecode


class PairwiseGpuSampler(BaseGpuSampler):

    def __init__(
        self, algo, n_parallel,
        set_cpu_affinity=True,
        cpu_assignments=None
    ):
        """
        :type algo: GpuBatchPolopt
        """
        self.algo = algo
        self.batch_size = self.algo.batch_size
        self.n_parallel = n_parallel
        self.set_cpu_affinity = set_cpu_affinity
        self.cpu_assignments = cpu_assignments
        self.initialized = False

    def initialize_par_objs(self):
        """ Build all parallel objects """
        n_par = self.n_parallel
        obs_shape, obs_size, obs_tc = get_space_properties(self.algo.env.observation_space)
        act_shape, act_size, act_tc = get_space_properties(self.algo.env.action_space)

        shutdown = mp.RawValue(c_bool, False)
        barrier = mp.Barrier(n_par + 1)
        # NOTE: using typecode 'f' for float32.
        n_arr = np.ctypeslib.as_array
        m_arr = mp.RawArray
        shareds = struct(
            obs=[n_arr(m_arr(obs_tc, n_par * obs_size)).reshape(n_par, *obs_shape)
                for _ in range(2)],
            act=[n_arr(m_arr(act_tc, n_par * act_size)).reshape(n_par, *act_shape)
                for _ in range(2)],
            rew=[n_arr(m_arr('f', n_par)) for _ in range(2)],
            done=[m_arr(c_bool, n_par) for _ in range(2)],
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
        self.env = (self.algo.env, copy.deepcopy(self.algo.env))

    @gt.wrap
    def obtain_samples(self, itr):
        """ Only master (GPU) thread executes this method """
        if self.set_cpu_affinity:
            self.par_objs.master_process.cpu_affinity([0])
        self.par_objs.barrier.wait()  # signal workers to re-enter

        shareds = self.par_objs.shareds
        act_waiters = self.par_objs.semaphores.act_waiters
        step_blockers = self.par_objs.semaphores.step_blockers
        obs_flatten_n = self.algo.env.observation_space.flatten_n
        act_flatten_n = self.algo.env.action_space.flatten_n
        get_actions = self.algo.policy.get_actions

        shareds.continue_sampling.value = True  # (set before workers check)
        continue_sampling = True

        paths = []
        cum_length_complete_paths = 0
        range_par = range(self.n_parallel)
        sims_act = [[[] for _ in range_par] for _ in range(2)]
        sims_obs = [[[] for _ in range_par] for _ in range(2)]
        sims_rew = [[[] for _ in range_par] for _ in range(2)]
        sims_agent_info = [[[] for _ in range_par] for _ in range(2)]
        all_obs = [None] * 2
        all_act = [None] * 2
        all_agent_info = [None] * 2
        all_rew = [None] * 2
        all_done = np.empty(len(shareds.done[0]), dtype='bool')  # buffer

        pbar = ProgBarCounter(self.batch_size)

        [b.acquire() for b in step_blockers[0]]

        # Start-up: do not record yet.
        next_obs = shareds.obs[0].copy()  # new object
        next_act, all_agent_info[0] = get_actions(next_obs)  # new objects
        shareds.act[0][:] = next_act
        all_obs[0] = obs_flatten_n(next_obs)
        all_act[0] = act_flatten_n(next_act)

        [w.release() for w in act_waiters[0]]
        [b.acquire() for b in step_blockers[1]]

        next_obs = shareds.obs[1].copy()  # new object
        next_act, all_agent_info[1] = get_actions(next_obs)  # new objects
        shareds.act[1][:] = next_act
        all_obs[1] = obs_flatten_n(next_obs)
        all_act[1] = act_flatten_n(next_act)

        [w.release() for w in act_waiters[1]]

        gt.stamp('startup')
        # loop = gt.timed_loop('samp', save_itrs=False)
        # samp_itr = -1
        j = 1
        while continue_sampling:
            # next(loop)
            # samp_itr += 1
            # time.sleep(0.01)
            j = j ^ 1  # xor -- toggles

            [b.acquire() for b in step_blockers[j]]

            all_rew[j] = shareds.rew[j].copy()  # new object
            next_obs = shareds.obs[j].copy()  # new object
            all_done[:] = shareds.done[j]  # copy to buffer

            # gt.stamp('copy')
            next_act, next_agent_info = get_actions(next_obs)  # new objects
            shareds.act[j][:] = next_act
            # gt.stamp('get_act')

            [w.release() for w in act_waiters[j]]

            # Move current data into persistent variables.
            for i in range_par:
                sims_act[j][i].append(all_act[j][i])
                sims_rew[j][i].append(all_rew[j][i])
                sims_obs[j][i].append(all_obs[j][i])
                sims_agent_info[j][i].append({k: v[i] for k, v in all_agent_info[j].items()})
                if all_done[i]:
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs[j][i]),
                        actions=tensor_utils.stack_tensor_list(sims_act[j][i]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew[j][i]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info[j][i]),
                        env_infos={},  # (have yet to see an env provide this)
                    ))
                    cum_length_complete_paths += len(sims_rew[j][i])
                    sims_act[j][i] = []
                    sims_obs[j][i] = []
                    sims_rew[j][i] = []
                    sims_agent_info[j][i] = []
            if any(all_done):
                continue_sampling = (cum_length_complete_paths < self.batch_size)
                pbar.update(cum_length_complete_paths)
            all_obs[j] = obs_flatten_n(next_obs)  # (flatten doesn't necessarily copy)
            all_act[j] = act_flatten_n(next_act)
            all_agent_info[j] = next_agent_info

        # loop.exit()
        gt.stamp('samp')

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

        # time.sleep(0.001)
        # print("Master exited sampling loop: ", itr, "at count: ", samp_itr)

        return paths

    @gt.wrap
    def obtain_samples_worker(self, rank, semaphores):
        """ Workers execute synchronously with master in obtain_samples() """

        shareds = self.par_objs.shareds
        step_blocker = semaphores.step_blocker  # (a pair)
        act_waiter = semaphores.act_waiter  # (a pair)
        env = self.env  # (a pair)

        # Start-up: provide first observations.
        shareds.obs[0][rank][:] = env[0].reset()
        step_blocker[0].release()
        shareds.obs[1][rank][:] = env[1].reset()
        step_blocker[1].release()
        gt.stamp('first_obs')
        act_waiter[0].acquire()
        gt.stamp('bar_first_act')

        # loop = gt.timed_loop('samp', save_itrs=False)
        # false_waiters = []
        # samp_itr = -1
        j = 0
        while shareds.continue_sampling.value:
            # next(loop)
            # Synchronization diagnostic / test:
            # samp_itr += 1
            # print(rank, samp_itr)
            # if rank == 0:
            #     time.sleep(0.01)

            o, r, d, env_info = env[j].step(shareds.act[j][rank])

            # TODO: Later, might want to have the simulator skip an iteration to reset.
            if d:
                o = env[j].reset()

            # Share all the current data.
            shareds.obs[j][rank][:] = o
            shareds.rew[j][rank] = r
            shareds.done[j][rank] = d
            # (have yet to see an environment provide env_info)

            step_blocker[j].release()
            j = j ^ 1  # xor -- toggles
            act_waiter[j].acquire()  # normal code, turn off for diagnostic
            # Synchronization diagnostic:
            # if samp_itr < 100:
            #     act_waiter[j].acquire()  # warmup
            # elif not act_waiter[j].acquire(block=False):
            #     false_waiters.append(samp_itr)
            #     act_waiter[j].acquire()

        # loop.exit()
        gt.stamp('samp', qp=False)

        # Every rank should print the same iteration count, 2 greater than the
        # iteration count of the master.
        # time.sleep(0.01 * rank)
        # print("Rank: ", rank, "exited sampling loop at count: ", samp_itr)
        # print("\nRank: ", rank, " loop count: ", samp_itr, "  num false_waiters: ", len(false_waiters))  # , " false_waiters: \n", false_waiters)

