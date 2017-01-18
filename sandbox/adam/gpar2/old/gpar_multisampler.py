from rllab.misc import tensor_utils
import numpy as np
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
from rllab.sampler.base import BaseSampler
# import copy
import gtimer as gt


class BarrierLite(object):
    """
    Much faster than multiprocessing.Barrier:

    WARNING:
    One of these by itself in a fast loop does not work!
    Need an alternating pair.
    """

    def __init__(self, n):
        # self.n = n
        self.range_n_1 = range(n - 1)
        # self.range_n_2 = range(n - 2)
        self.semaphore_1 = mp.Semaphore(n - 1)
        self.semaphore_2 = mp.Semaphore(0)

    # @gt.wrap(save_itrs=False)
    def wait(self):
        # gt.stamp('enter')
        if self.semaphore_1.acquire(block=False):
            self.semaphore_2.acquire()
        else:
            [self.semaphore_1.release() for _ in self.range_n_1]
            [self.semaphore_2.release() for _ in self.range_n_1]
        # gt.stamp('exit')

    # def wait_switch(self, switch):
    #     if switch:
    #         if self.semaphore_1[0].acquire(block=False):
    #             self.semaphore_2[0].acquire()
    #         else:
    #             [self.semaphore_1[0].release() for _ in self.range_n_1]
    #             [self.semaphore_2[0].release() for _ in self.range_n_1]
    #     else:
    #         if self.semaphore_1[1].acquire(block=False):
    #             self.semaphore_2[1].acquire()
    #         else:
    #             [self.semaphore_1[1].release() for _ in self.range_n_1]
    #             [self.semaphore_2[1].release() for _ in self.range_n_1]


class SingletonBarrier(object):
    """ Meaning: one waits for all others. """

    def __init__(self, n):
        self.range_release = range(n - 1)
        self.signaler = mp.Semaphore(n - 1)
        self.waiter = mp.Semaphore(0)

    def wait(self):
        self.waiter.acquire()

    def checkin(self):
        if not self.signaler.acquire(block=False):
            self.waiter.release()
            [self.signaler.release() for _ in self.range_release]


class MultiplicitousBarrier(object):
    """ Meaning: all other wait for one. """

    def __init__(self, n):
        self.range_release = range(n)
        self.waiters = mp.Semaphore(0)

    def wait(self):
        self.waiters.acquire()

    def release(self):
        [self.waiters.release() for _ in self.range_release]


class GParMultiSampler(BaseSampler):

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
        self.init_par_objs(self.n_parallel, self.n_simulators, obs_dim, act_dim, state_info_keys)  # (before processes fork)

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def init_par_objs(self, n_par, n_sim, obs_dim, act_dim, state_info_keys):
        """ Before processes fork, build shared variables and synchronizers """
        # NOTE: using typecode 'f' for float32.
        n = n_par * n_sim
        shareds = struct(
            current_obs=np.ctypeslib.as_array(
                mp.RawArray('f', n * obs_dim)).reshape(n, obs_dim),
            current_act=np.ctypeslib.as_array(
                mp.RawArray('f', n * act_dim)).reshape(n, act_dim),
            current_rew=np.ctypeslib.as_array(mp.RawArray('f', n)),
            current_done=mp.RawArray(c_bool, n),
            continue_sampling=mp.RawValue(c_bool),
        )
        for k in state_info_keys:
            shareds.k = np.ctypeslib.as_array(mp.RawArray('f', n))  # Assume scalar for now
        barriers = struct(
            # step_done=SingletonBarrier(n),
            # act_ready=MultiplicitousBarrier(n),
            # first_step_done=SingletonBarrier(n),
            # first_act_ready=MultiplicitousBarrier(n),
            # step_done=BarrierLite(n + 1),  # +1 for master process.
            # act_ready=BarrierLite(n + 1),
            step_done=mp.Barrier(n_par + 1),
            act_ready=mp.Barrier(n_par + 1),
            enter_obtain=mp.Barrier(n_par + 1),
        )
        self.par_objs = struct(shareds=shareds, barriers=barriers)

    @gt.wrap
    def obtain_samples_master(self, itr):
        print("Master enter obtain itr: ", itr)
        par = self.par_objs
        par.shareds.continue_sampling.value = True
        continue_sampling = True
        par.barriers.enter_obtain.wait()
        print("Master past barrier enter obtain itr: ", itr)
        range_idx = range(self.n_parallel * self.n_simulators)

        paths = []
        cum_length_complete_paths = 0
        sims_act = [[] for _ in range_idx]
        sims_obs = [[] for _ in range_idx]
        sims_rew = [[] for _ in range_idx]

        # Start-up: do not record yet.
        # par.barriers.first_step_done.wait()  # initial observations ready
        par.barriers.step_done.wait()
        gt.stamp('bar_first_step')
        all_obs = par.shareds.current_obs.copy()
        all_act, all_agent_info = self.algo.policy.get_actions(all_obs)  # could instead send shareds.current_obs
        # agent_info_blank = {k: [] for k in all_agent_info.keys()}
        # sims_agent_info = [copy.copy(agent_info_blank) for _ in range_idx]
        sims_agent_info = [[] for _ in range_idx]
        par.shareds.current_act[:] = all_act  # copy just for building the paths.
        gt.stamp('first_act')
        # par.barriers.act_ready.release()
        par.barriers.act_ready.wait()
        gt.stamp('bar_first_act')

        print("Master starting sampling loop: ", itr)
        # loop = gt.timed_loop('samp', save_itrs=False)
        i = 0
        while continue_sampling:
            i += 1
            # next(loop)
            # Simulators write next observation and reward.
            par.barriers.step_done.wait()
            # gt.stamp('bar_step')

            all_rew = par.shareds.current_rew.copy()
            next_obs = par.shareds.current_obs.copy()
            all_done = par.shareds.current_done[:]
            # gt.stamp('copy')
            next_act, next_agent_info = self.algo.policy.get_actions(next_obs)
            # gt.stamp('get_act')
            par.shareds.current_act[:] = next_act
            # gt.stamp('write_act')

            # par.barriers.act_ready.release()
            par.barriers.act_ready.wait()
            # gt.stamp('bar_act')

            # All of the below happens while the simulators are stepping.

            # Move current data into persistent variables.
            # TODO: think about how to copy just into final concatenated arrays?
            # No, can't do that, because...oh actually can do that, just the paths
            # will be interleaved with each other, and so I'll have to keep track
            # of the order for writing the advantages, as well.  Eh maybe this
            # existing way will be fast enough anyway.
            for idx in range(self.n_parallel * self.n_simulators):
                sims_act[idx].append(all_act[idx, :])  # use new 'all_act' each iteration instead of shared_act, to make sure it's not just a view to shared variable, which will change values
                sims_rew[idx].append(all_rew[idx])
                sims_obs[idx].append(all_obs[idx, :])
                sims_agent_info[idx].append({k: v[idx] for k, v in all_agent_info.items()})
                # for k in all_agent_info:
                #     sims_agent_info[idx][k].append(all_agent_info[k][idx])
                if all_done[idx]:
                    # print(sims_agent_info[idx])
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs[idx]),
                        actions=tensor_utils.stack_tensor_list(sims_act[idx]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew[idx]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info[idx]),
                        env_infos={},
                    ))
                    cum_length_complete_paths += len(sims_rew[idx])
                    # Must set this condition before action barrier, so simulators get it.
                    continue_sampling = (cum_length_complete_paths < self.batch_size)
                    sims_act[idx] = []
                    sims_obs[idx] = []
                    sims_rew[idx] = []
                    # sims_agent_info[idx] = copy.copy(agent_info_blank)
                    sims_agent_info[idx] = []
            all_obs = next_obs
            all_act = next_act
            all_agent_info = next_agent_info

            # BUT NOW NEED TO CHANGE THE SIGNAL TO EXIT THE LOOP.

            # all_obs = par.shareds.current_obs.copy()  # (append next time)
            # gt.stamp('store')

            # # Get the actions for all simulators at once.
            # all_act, all_agent_info = self.algo.policy.get_actions(all_obs)  # or write direct to shared if I change copy scheme later.
            # par.shareds.current_act[:] = all_act
            # gt.stamp('get_act')

            # # Master done writing actions and capturing current data.
            # par.barriers.act_ready.release()
            # gt.stamp('bar_act')
        # loop.exit()
        print("Master exited sampling loop: ", itr, "at count: ", i)
        gt.stamp('samp')

        # Have to go through these one more time for the simulators to get
        # the signal to leave the while loop.
        # (Could change the simulator loop to avoid this, but it's nice to have
        # the first barriers out front to see the time between iterations.)
        par.barriers.step_done.wait()
        par.shareds.continue_sampling.value = False
        # par.barriers.act_ready.release()
        par.barriers.act_ready.wait()

        return paths

    @gt.wrap
    def obtain_samples_simulator(self, rank, itr):
        print("Rank: ", rank, "entering obtain itr: ", itr)
        par = self.par_objs
        n_par = self.n_parallel

        # Start-up: provide first observation.
        for i, env in enumerate(self.algo.envs):
            par.shareds.current_obs[rank + i * n_par, :] = env.reset()

        par.barriers.enter_obtain.wait()
        print("Rank: ", rank, "past barrier enter obtain itr: ", itr)
        # par.barriers.step_done.checkin()
        par.barriers.step_done.wait()
        gt.stamp('bar_first_step')
        # par.barriers.first_act_ready.wait()
        par.barriers.act_ready.wait()
        gt.stamp('bar_first_act')

        print("Rank: ", rank, "starting sampling loop itr: ", itr)
        # print("Rank: ", rank, "starting loop condition: ", par.shareds.continue_sampling.value)
        # loop = gt.timed_loop('samp', save_itrs=False)
        j = 0
        while par.shareds.continue_sampling.value:
            j += 1
            # next(loop)
            # Step the simulator.
            for i, env in enumerate(self.algo.envs):
                idx = rank + i * n_par
                o, r, d, env_info = env.step(par.shareds.current_act[idx, :])
                # gt.stamp('step')
                # TODO: Later, might want to have the simulator skip an iteration to reset.
                if d:
                    o = env.reset()
                    # gt.stamp('reset')

                # Write all the current data.
                par.shareds.current_obs[idx, :] = o
                par.shareds.current_rew[idx] = r
                par.shareds.current_done[idx] = d
            # Do nothing with env_info yet.
            # gt.stamp('share')
            # Signal that writing is complete.
            # par.barriers.step_done.checkin()
            par.barriers.step_done.wait()
            # gt.stamp('bar_step')
            # Wait for master to return action or loop condition.
            # NOTE: evaluate loop condition after this barrier.
            par.barriers.act_ready.wait()
            # gt.stamp('bar_act')
        # loop.exit()
        print("Rank: ", rank, "exited sampling loop: ", itr, "at count: ", j)
        gt.stamp('samp', qp=True)
