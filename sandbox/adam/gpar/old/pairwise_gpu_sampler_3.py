from rllab.misc import tensor_utils
import numpy as np
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
from rllab.sampler.base import BaseSampler
from rllab.misc import logger
import pyprind
# import copy
import gtimer as gt
import time


class PairwiseGpuSampler_3(BaseSampler):

    def __init__(self, algo):
        """
        :type algo: ParallelGpuBatchPolopt
        """
        self.algo = algo
        self.n_parallel = self.algo.n_parallel
        self.n_simulators = self.algo.n_simulators
        self.batch_size = self.algo.batch_size

        state_info_keys = self.algo.policy.state_info_keys
        obs_dim = int(self.algo.env.observation_space.flat_dim)
        act_dim = int(self.algo.env.action_space.flat_dim)
        self.init_par_objs(self.n_parallel, self.n_simulators,
            obs_dim, act_dim, state_info_keys)  # (before processes fork)

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def init_par_objs(self, n_par, n_sim, obs_dim, act_dim, state_info_keys):
        """ Before processes fork, build shared variables and synchronizers """
        # NOTE: using typecode 'f' for float32.
        n = n_par * n_sim
        shareds = struct(
            obs=[np.ctypeslib.as_array(
                mp.RawArray('f', n * obs_dim)).reshape(n, obs_dim)
                for _ in range(2)],
            act=[np.ctypeslib.as_array(
                mp.RawArray('f', n * act_dim)).reshape(n, act_dim)
                for _ in range(2)],
            rew=[np.ctypeslib.as_array(mp.RawArray('f', n))
                for _ in range(2)],
            done=[mp.RawArray(c_bool, n) for _ in range(2)],
            continue_sampling=mp.RawValue(c_bool, True),
        )
        for k in state_info_keys:
            shareds[k] = [np.ctypeslib.as_array(mp.RawArray('f', n))
                for _ in range(2)]  # Assume scalar for now
        semaphores = struct(
            step_waiter=(mp.Semaphore(0), mp.Semaphore(0)),
            step_blocker=(mp.Semaphore(n_par - 1), mp.Semaphore(n_par - 1)),
            act_waiters=([mp.Semaphore(0) for _ in range(n_par)],
                         [mp.Semaphore(0) for _ in range(n_par)]),
        )
        # gates = struct(
        #     step=(ManyBlocksOneGate(n_par), ManyBlocksOneGate(n_par)),
        #     act=(OneBlocksManyGate(n_par), OneBlocksManyGate(n_par)),
        # )
        self.par_objs = struct(shareds=shareds, semaphores=semaphores)

    @gt.wrap
    def obtain_samples_master(self, itr):
        par = self.par_objs

        act_waiters = par.semaphores.act_waiters
        step_waiter = par.semaphores.step_waiter

        # Setting the shared value here works because simulators wait at first
        # act_gate before entering the while loop.
        par.shareds.continue_sampling.value = True
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

        step_waiter[0].acquire()  # gates.step[0].wait()

        # Start-up: do not record yet.
        all_obs[0] = par.shareds.obs[0].copy()
        all_act[0], all_agent_info[0] = self.algo.policy.get_actions(all_obs[0])  # could instead send shareds.current_obs
        par.shareds.act[0][:] = all_act[0]  # copy just for building the paths.

        [w.release() for w in act_waiters[0]]  # gates.act[0].open()

        step_waiter[1].acquire()  # gates.step[1].wait()

        all_obs[1] = par.shareds.obs[1].copy()
        all_act[1], all_agent_info[1] = self.algo.policy.get_actions(all_obs[1])
        par.shareds.act[1][:] = all_act[1]

        [waiter.release() for waiter in act_waiters[1]]  # gates.act[1].open()

        gt.stamp('startup')
        # loop = gt.timed_loop('samp', save_itrs=False)
        # i = -1
        j = 1
        while continue_sampling:
            # next(loop)
            # i += 1
            j = j ^ 1  # xor -- toggles
            # time.sleep(0.01)

            step_waiter[j].acquire()  # gates.step[j].wait()

            all_rew[j] = par.shareds.rew[j].copy()
            next_obs[j] = par.shareds.obs[j].copy()
            all_done[j] = par.shareds.done[j][:]

            # gt.stamp('copy')
            par.shareds.act[j][:], next_agent_info[j] = \
                self.algo.policy.get_actions(next_obs[j])
            # gt.stamp('get_act')

            [w.release() for w in act_waiters[j]]  # gates.act[j].open()

            # Move current data into persistent variables.
            # TODO: think about how to copy just into final concatenated arrays?
            # No, can't do that, because...oh actually can do that, just the paths
            # will be interleaved with each other, and so I'll have to keep track
            # of the order for writing the advantages, as well.  Eh maybe this
            # existing way will be fast enough anyway.
            for idx in range_sims:
                sims_act[j][idx].append(all_act[j][idx, :])  # use new 'all_act' each iteration instead of shared_act, to make sure it's not just a view to shared variable, which will change values
                sims_rew[j][idx].append(all_rew[j][idx])
                sims_obs[j][idx].append(all_obs[j][idx, :])
                sims_agent_info[j][idx].append({k: v[idx] for k, v in all_agent_info[j].items()})
                if all_done[j][idx]:
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs[j][idx]),
                        actions=tensor_utils.stack_tensor_list(sims_act[j][idx]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew[j][idx]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info[j][idx]),
                        env_infos={},
                    ))
                    cum_length_complete_paths += len(sims_rew[j][idx])
                    continue_sampling = (cum_length_complete_paths < self.batch_size)
                    sims_act[j][idx] = []
                    sims_obs[j][idx] = []
                    sims_rew[j][idx] = []
                    sims_agent_info[j][idx] = []
            if any(all_done[j]):
                pbar.update(cum_length_complete_paths)
            all_obs[j] = next_obs[j]
            all_act[j] = par.shareds.act[j][:]
            all_agent_info[j] = next_agent_info[j]

        # loop.exit()
        gt.stamp('samp')
        # print("Master exited sampling loop: ", itr, "at count: ", i)

        # Simulators do 2 more steps, since the act_gates have already been
        # opened.  Wait for them to do the 2nd step (the same j) and get stopped
        # at the next act_gate.
        step_waiter[j].acquire()  # gates.step[j].wait()
        # Now the simulators are waiting at the action of (j ^ 1)
        par.shareds.continue_sampling.value = False
        j = j ^ 1
        [w.release() for w in act_waiters[j]]  # gates.act[j].open()
        # Now the simulators check the while condition and exit.
        # Just to close the gate, (simulators opened it):
        step_waiter[j].acquire()  # gates.step[j].wait()
        pbar.stop()

        return paths

    @gt.wrap
    def obtain_samples_simulator(self, rank, itr):
        par = self.par_objs

        step_blocker = par.semaphores.step_blocker
        step_waiter = par.semaphores.step_waiter
        act_waiter = (par.semaphores.act_waiters[0][rank],
                      par.semaphores.act_waiters[1][rank])
        release_range = range(self.n_parallel - 1)

        # Assign a block of rows to each rank.
        start_idx = rank * self.n_simulators
        idx = [start_idx + s for s in range(self.n_simulators)]

        # Start-up: provide first observations.
        for s, env in enumerate(self.algo.envs[0]):
            par.shareds.obs[0][idx[s], :] = env.reset()
        # par.shareds.obs[0][rank, :] = self.algo.env[0].reset()
        if not step_blocker[0].acquire(block=False):  # gates.step[0].checkin()
            step_waiter[0].release()
            [step_blocker[0].release() for _ in release_range]

        for s, env in enumerate(self.algo.envs[1]):
            par.shareds.obs[1][idx[s], :] = env.reset()
        # par.shareds.obs[1][rank, :] = self.algo.env[1].reset()
        if not step_blocker[1].acquire(block=False):  # gates.step[1].checkin()
            step_waiter[1].release()
            [step_blocker[1].release() for _ in release_range]

        # Waiting for first act before the loop allows the master to reset the
        # while-loop condition, and waiting for an act at the end of the loop
        # simplifies exiting.
        act_waiter[0].acquire()  # gates.act[0].wait()
        gt.stamp('bar_first_act')

        # loop = gt.timed_loop('samp', save_itrs=False)
        # i = -1
        j = 0
        while par.shareds.continue_sampling.value:
            # next(loop)
            # Synchronization diagnostic / test:
            # i += 1
            # print(rank, i)
            # if rank == 0:
            #     time.sleep(0.01)

            for s, env in enumerate(self.algo.envs[j]):
                o, r, d, env_info = env.step(par.shareds.act[j][idx[s], :])
                # gt.stamp('step')
                # TODO: Later, might want to have the simulator skip an iteration to reset.
                if d:
                    o = env.reset()
                    # gt.stamp('reset')

                # Share all the current data.
                par.shareds.obs[j][idx[s], :] = o
                par.shareds.rew[j][idx[s]] = r
                par.shareds.done[j][idx[s]] = d

            if not step_blocker[j].acquire(block=False):  # gates.step[j].checkin()
                step_waiter[j].release()
                [step_blocker[j].release() for _ in release_range]
            j = j ^ 1  # xor -- toggles
            act_waiter[j].acquire()  # act_gate[j].wait(rank)

            # gt.stamp('bar_act')
        # loop.exit()
        gt.stamp('samp', qp=False)
        # Every rank should print the same iteration count, 2 greater than the
        # iteration count of the master.
        # print("Rank: ", rank, "exited sampling loop: ", itr, "at count: ", i)


class ProgBarCounter(object):
    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def update(self, current_count):
        if not logger.get_log_tabular_only():
            self.cur_count = current_count
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        if self.pbar is not None and self.pbar.active:
            self.pbar.stop()




################################################################################
# These must be used as a pair, in fact in two alternating pairs.              #
#                                                                              #
# Class definitions just included here for clarification, but execution is     #
# faster when functionality written directly into code rather than wrapping in #
# classes.                                                                     #
################################################################################

# class ManyBlocksOneGate(object):
#     """ Special one waits on many blockers """

#     def __init__(self, n_blockers):
#         self.range_release = range(n_blockers - 1)
#         self.waiter = mp.Semaphore(0)
#         self.blocker = mp.Semaphore(n_blockers - 1)

#     def wait(self):
#         self.waiter.acquire()

#     def checkin(self):
#         if not self.blocker.acquire(block=False):
#             self.waiter.release()  # open the gate
#             [self.blocker.release() for _ in self.range_release]


# class OneBlocksManyGate(object):
#     """ All others wait for one blocker """

#     def __init__(self, n_waiters):
#         self.waiters = [mp.Semaphore(0) for _ in range(n_waiters)]

#     def wait(self, rank):
#         self.waiters[rank].acquire()

#     def open(self):
#         [waiter.release() for waiter in self.waiters]

################################################################################
################################################################################
