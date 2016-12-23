from rllab.misc import tensor_utils
import numpy as np
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
from rllab.sampler.base import BaseSampler
# import copy
import gtimer as gt
import time

class FastBarrier(object):
    """
    WARNING: Not safe to use one of these in a loop, must use an alternating pair.
    """

    def __init__(self, n):
        self.range_release = range(n - 1)
        self.n = n
        self.enter = mp.Semaphore(n - 1)
        self.hold = mp.Semaphore(0)

    def wait(self):
        if self.enter.acquire(block=False):
            self.hold.acquire()
        else:
            [self.enter.release() for _ in range(self.n - 1)]
            [self.hold.release() for _ in range(self.n - 1)]


#################################################################
# It's not safe to use a single pair of these gates in a loop.  #
# Must use two pairs and alternate them.                        #
# That's no longer true!! One pair of these does the job!       #
#################################################################

class ManyBlocksOneGate(object):
    """ Special one waits on many blockers """

    def __init__(self, n_blockers):
        self.range_release = range(n_blockers - 1)
        self.waiter = mp.Semaphore(0)
        self.blocker = mp.Semaphore(n_blockers - 1)

    def wait(self):
        self.waiter.acquire()

    def checkin(self):
        if not self.blocker.acquire(block=False):
            self.waiter.release()
            [self.blocker.release() for _ in self.range_release]


class OneBlocksManyGate(object):
    """ All others wait for one blocker """

    def __init__(self, n_waiters):
        self.waiters = [mp.Semaphore(0) for _ in range(n_waiters)]

    def wait(self, rank):
        self.waiters[rank].acquire()

    def release(self):
        [waiter.release() for waiter in self.waiters]

####################################################################
####################################################################



class PairwiseGpuSampler(BaseSampler):

    def __init__(self, algo):
        """
        :type algo: ParallelGpuBatchPolopt
        """
        self.algo = algo
        self.n_parallel = self.algo.n_parallel
        self.batch_size = self.algo.batch_size

        state_info_keys = self.algo.policy.state_info_keys
        obs_dim = int(self.algo.env.observation_space.flat_dim)
        act_dim = int(self.algo.env.action_space.flat_dim)
        self.init_par_objs(self.n_parallel, obs_dim, act_dim, state_info_keys)  # (before processes fork)

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def init_par_objs(self, n_par, obs_dim, act_dim, state_info_keys):
        """ Before processes fork, build shared variables and synchronizers """
        # NOTE: using typecode 'f' for float32.
        shareds = struct(
            current_obs_a=np.ctypeslib.as_array(
                mp.RawArray('f', n_par * obs_dim)).reshape(n_par, obs_dim),
            current_obs_b=np.ctypeslib.as_array(
                mp.RawArray('f', n_par * obs_dim)).reshape(n_par, obs_dim),
            current_act_a=np.ctypeslib.as_array(
                mp.RawArray('f', n_par * act_dim)).reshape(n_par, act_dim),
            current_act_b=np.ctypeslib.as_array(
                mp.RawArray('f', n_par * act_dim)).reshape(n_par, act_dim),
            current_rew_a=np.ctypeslib.as_array(mp.RawArray('f', n_par)),
            current_rew_b=np.ctypeslib.as_array(mp.RawArray('f', n_par)),
            current_done_a=mp.RawArray(c_bool, n_par),
            current_done_b=mp.RawArray(c_bool, n_par),
            continue_sampling=mp.RawValue(c_bool),
        )
        for k in state_info_keys:
            shareds[k + '_a'] = np.ctypeslib.as_array(mp.RawArray('f', n_par))  # Assume scalar for now
            shareds[k + '_b'] = np.ctypeslib.as_array(mp.RawArray('f', n_par))
        barriers = struct(
            first_obs=mp.Barrier(n_par + 1),  # +1 for master process.
            # first_act=mp.Barrier(n_par + 1),
            # step_A_act_B=mp.Barrier(n_par + 1),
            # step_B_act_A=mp.Barrier(n_par + 1),
            # step_A_act_B=FastBarrier(n_par + 1),
            # step_B_act_A=FastBarrier(n_par + 1),
            # step_A_act_B_enter=mp.Semaphore(n_par),
            # step_A_act_B_hold=mp.Semaphore(0),
            # step_B_act_A_enter=mp.Semaphore(n_par),
            # step_B_act_A_hold=mp.Semaphore(0),
            # step_A=ManyBlocksOneGate(n_par),
            # act_A=OneBlocksManyGate(n_par),
            # step_B=ManyBlocksOneGate(n_par),
            # act_B=OneBlocksManyGate(n_par),
            step_A_waiter=mp.Semaphore(0),
            step_A_blocker=mp.Semaphore(n_par - 1),
            step_B_waiter=mp.Semaphore(0),
            step_B_blocker=mp.Semaphore(n_par - 1),
            act_A_waiters=[mp.Semaphore(0) for _ in range(n_par)],
            act_B_waiters=[mp.Semaphore(0) for _ in range(n_par)],

            enter_obtain=mp.Barrier(n_par + 1),
        )
        self.par_objs = struct(shareds=shareds, barriers=barriers)
        self.release_range = range(n_par)
        self.release_range_m1 = range(n_par - 1)

    @gt.wrap
    def obtain_samples_master(self, itr):
        print("Master enter obtain itr: ", itr)
        par = self.par_objs
        act_A_waiters = par.barriers.act_A_waiters
        act_B_waiters = par.barriers.act_B_waiters
        step_A_waiter = par.barriers.step_A_waiter
        step_B_waiter = par.barriers.step_B_waiter
        # release_range = self.release_range

        # step_A = par.barriers.step_A
        # step_B = par.barriers.step_B
        # act_A = par.barriers.act_A
        # act_B = par.barriers.act_B

        par.shareds.continue_sampling.value = True
        continue_sampling = True
        print("Master past barrier enter obtain itr: ", itr)
        range_par = range(self.n_parallel)

        paths = []
        cum_length_complete_paths = 0
        sims_act_a = [[] for _ in range_par]
        sims_obs_a = [[] for _ in range_par]
        sims_rew_a = [[] for _ in range_par]
        sims_agent_info_a = [[] for _ in range_par]
        sims_act_b = [[] for _ in range_par]
        sims_obs_b = [[] for _ in range_par]
        sims_rew_b = [[] for _ in range_par]
        sims_agent_info_b = [[] for _ in range_par]

        # par.barriers.first_obs.wait()

        # step_A.wait()
        step_A_waiter.acquire()

        # par.barriers.first_obs.wait()  # First obs for both 'a' and 'b' written before this passes.
        # Start-up: do not record yet.
        all_obs_a = par.shareds.current_obs_a.copy()
        all_act_a, all_agent_info_a = self.algo.policy.get_actions(all_obs_a)  # could instead send shareds.current_obs
        par.shareds.current_act_a[:] = all_act_a  # copy just for building the paths.

        # act_A.release()
        [waiter.release() for waiter in act_A_waiters]

        # step_B.wait()
        step_B_waiter.acquire()

        all_obs_b = par.shareds.current_obs_b.copy()
        all_act_b, all_agent_info_b = self.algo.policy.get_actions(all_obs_b)
        par.shareds.current_act_b[:] = all_act_b

        # par.barriers.first_act.wait()

        # act_A.release()
        # [act_A_waiter.release() for _ in release_range]

        # act_B.release()
        [waiter.release() for waiter in act_B_waiters]

        par.barriers.first_obs.wait()

        gt.stamp('prep')

        print("Master starting sampling loop: ", itr)
        # loop = gt.timed_loop('samp', save_itrs=False)
        i = 0
        while continue_sampling:
            i += 1
            # next(loop)
            # time.sleep(0.01)
            # GROUP A.

            # step_A.wait()
            step_A_waiter.acquire()

            all_rew_a = par.shareds.current_rew_a.copy()
            next_obs_a = par.shareds.current_obs_a.copy()
            all_done_a = par.shareds.current_done_a[:]

            # gt.stamp('copy')
            next_act_a, next_agent_info_a = self.algo.policy.get_actions(next_obs_a)
            # gt.stamp('get_act')
            par.shareds.current_act_a[:] = next_act_a
            # gt.stamp('write_act')

            # act_A.release()
            [waiter.release() for waiter in act_A_waiters]

            # Move current data into persistent variables.
            # TODO: think about how to copy just into final concatenated arrays?
            # No, can't do that, because...oh actually can do that, just the paths
            # will be interleaved with each other, and so I'll have to keep track
            # of the order for writing the advantages, as well.  Eh maybe this
            # existing way will be fast enough anyway.
            for rank in range_par:
                sims_act_a[rank].append(all_act_a[rank, :])  # use new 'all_act' each iteration instead of shared_act, to make sure it's not just a view to shared variable, which will change values
                sims_rew_a[rank].append(all_rew_a[rank])
                sims_obs_a[rank].append(all_obs_a[rank, :])
                sims_agent_info_a[rank].append({k: v[rank] for k, v in all_agent_info_a.items()})
                if all_done_a[rank]:
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs_a[rank]),
                        actions=tensor_utils.stack_tensor_list(sims_act_a[rank]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew_a[rank]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info_a[rank]),
                        env_infos={},
                    ))
                    cum_length_complete_paths += len(sims_rew_a[rank])
                    continue_sampling = (cum_length_complete_paths < self.batch_size)
                    sims_act_a[rank] = []
                    sims_obs_a[rank] = []
                    sims_rew_a[rank] = []
                    sims_agent_info_a[rank] = []
            all_obs_a = next_obs_a
            all_act_a = next_act_a
            all_agent_info_a = next_agent_info_a

            # step_B.wait()
            step_B_waiter.acquire()


            # GROUP B.

            all_rew_b = par.shareds.current_rew_b.copy()
            next_obs_b = par.shareds.current_obs_b.copy()
            all_done_b = par.shareds.current_done_b[:]

            # gt.stamp('copy')
            next_act_b, next_agent_info_b = self.algo.policy.get_actions(next_obs_b)
            # gt.stamp('get_act')
            par.shareds.current_act_b[:] = next_act_b
            # gt.stamp('write_act')

            # act_B.release()
            [waiter.release() for waiter in act_B_waiters]

            # Move current data into persistent variables.
            # TODO: think about how to copy just into final concatenated arrays?
            # No, can't do that, because...oh actually can do that, just the paths
            # will be interleaved with each other, and so I'll have to keep track
            # of the order for writing the advantages, as well.  Eh maybe this
            # existing way will be fast enough anyway.
            for rank in range_par:
                sims_act_b[rank].append(all_act_b[rank, :])  # use new 'all_act' each iteration instead of shared_act, to make sure it's not just a view to shared variable, which will change values
                sims_rew_b[rank].append(all_rew_b[rank])
                sims_obs_b[rank].append(all_obs_b[rank, :])
                sims_agent_info_b[rank].append({k: v[rank] for k, v in all_agent_info_b.items()})
                if all_done_b[rank]:
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs_b[rank]),
                        actions=tensor_utils.stack_tensor_list(sims_act_b[rank]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew_b[rank]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info_b[rank]),
                        env_infos={},
                    ))
                    cum_length_complete_paths += len(sims_rew_b[rank])
                    continue_sampling = (cum_length_complete_paths < self.batch_size)
                    sims_act_b[rank] = []
                    sims_obs_b[rank] = []
                    sims_rew_b[rank] = []
                    sims_agent_info_b[rank] = []
            all_obs_b = next_obs_b
            all_act_b = next_act_b
            all_agent_info_b = next_agent_info_b

        # loop.exit()
        gt.stamp('samp')
        print("Master exited sampling loop: ", itr, "at count: ", i)

        # Have to go through these one more time for the simulators to get
        # the signal to leave the while loop.

        # step_A.wait()
        step_A_waiter.acquire()
        # step_B.wait()
        step_B_waiter.acquire()
        # act_A.release()
        [waiter.release() for waiter in act_A_waiters]  # Allow sims to finish loop.
        # step_A.wait()
        step_A_waiter.acquire()
        par.shareds.continue_sampling.value = False  # Signal to break.
        # act_B.release()
        [waiter.release() for waiter in act_B_waiters]  # Allow sims to finish loop.
        # step_B.wait()
        step_B_waiter.acquire()

        # time.sleep(1)

        # x = step_A_waiter.acquire(block=False)
        # print("\nstep_A_waiter.acquire: ", x)
        # x = step_B_waiter.acquire(block=False)
        # print("\nstep_B_waiter.acquire: ", x)
        # i = 0
        # for _ in range(2 * self.n_parallel + 2):
        #     if act_A_waiter.acquire(block=False):
        #         i += 1
        #     else:
        #         [act_A_waiter.release() for _ in range(i)]
        #         break
        # print("\nAcquired Act_A this many times: ", i)
        # j = 0
        # for _ in range(2 * self.n_parallel + 2):
        #     if act_B_waiter.acquire(block=False):
        #         j += 1
        #     else:
        #         [act_B_waiter.release() for _ in range(j)]
        #         break
        # print("\nAcquired Act_B this many times: ", j)




        # # act_B.release()
        # [act_B_waiter.release() for _ in release_range]  # Allow sims to finish loop.


        ### THIS IS NOT FUNCTIONING!! ##

        return paths

    @gt.wrap
    def obtain_samples_simulator(self, rank, itr):
        print("Rank: ", rank, "entering obtain itr: ", itr)
        par = self.par_objs

        step_A_blocker = par.barriers.step_A_blocker
        step_A_waiter = par.barriers.step_A_waiter
        step_B_blocker = par.barriers.step_B_blocker
        step_B_waiter = par.barriers.step_B_waiter
        act_A_waiter = par.barriers.act_A_waiters[rank]
        act_B_waiter = par.barriers.act_B_waiters[rank]
        release_range_m1 = self.release_range_m1

        # step_A = par.barriers.step_A
        # step_B = par.barriers.step_B
        # act_A = par.barriers.act_A
        # act_B = par.barriers.act_B

        # Start-up: provide first observation.
        par.shareds.current_obs_a[rank, :] = self.algo.env_a.reset()
        # step_A.checkin()
        if not step_A_blocker.acquire(block=False):
            step_A_waiter.release()
            [step_A_blocker.release() for _ in release_range_m1]

        par.shareds.current_obs_b[rank, :] = self.algo.env_b.reset()
        # par.barriers.first_obs.wait()
        # # step_B.checkin()
        if not step_B_blocker.acquire(block=False):
            step_B_waiter.release()
            [step_B_blocker.release() for _ in release_range_m1]

        print("Rank: ", rank, "past barrier enter obtain itr: ", itr)
        # par.barriers.step_done.wait()
        gt.stamp('bar_first_step')

        # act_A.wait()
        # act_A_waiter.acquire()

        # par.barriers.first_act.wait()
        par.barriers.first_obs.wait()

        gt.stamp('bar_first_act')
        print("Rank: ", rank, "starting sampling loop itr: ", itr)


        # loop = gt.timed_loop('samp', save_itrs=False)
        i = 0
        while par.shareds.continue_sampling.value:
            i += 1
            # if rank == 0:
            #     time.sleep(0.001)

            # act_A.wait(rank)
            act_A_waiter.acquire()

            # next(loop)
            # Step the simulator 'a'.
            o, r, d, env_info = self.algo.env_a.step(par.shareds.current_act_a[rank, :])
            # gt.stamp('step')
            # TODO: Later, might want to have the simulator skip an iteration to reset.
            if d:
                o = self.algo.env_a.reset()
                # gt.stamp('reset')

            # Write all the current data.
            par.shareds.current_obs_a[rank, :] = o
            par.shareds.current_rew_a[rank] = r
            par.shareds.current_done_a[rank] = d

            # step_A.checkin()
            if not step_A_blocker.acquire(block=False):
                step_A_waiter.release()
                [step_A_blocker.release() for _ in release_range_m1]

            # act_B.wait(rank)
            act_B_waiter.acquire()

            # Step the simulator 'b'.
            o, r, d, env_info = self.algo.env_b.step(par.shareds.current_act_b[rank, :])
            # gt.stamp('step')
            # TODO: Later, might want to have the simulator skip an iteration to reset.
            if d:
                o = self.algo.env_b.reset()
                # gt.stamp('reset')

            # Write all the current data.
            par.shareds.current_obs_b[rank, :] = o
            par.shareds.current_rew_b[rank] = r
            par.shareds.current_done_b[rank] = d


            # step_B.checkin()
            if not step_B_blocker.acquire(block=False):
                step_B_waiter.release()
                [step_B_blocker.release() for _ in release_range_m1]




            # gt.stamp('bar_act')
        # loop.exit()
        print("Rank: ", rank, "exited sampling loop: ", itr, "at count: ", i)
        gt.stamp('samp', qp=False)

        # if rank == 0:
        #     time.sleep(1)
        #     i = 0
        #     for _ in range(self.n_parallel * 2 + 2):
        #         if step_A_blocker.acquire(block=False):
        #             i += 1
        #         else:
        #             [step_A_blocker.release() for _ in range(i)]
        #             break
        #     print("Acquired Step_A_blocker this many times: ", i)
        #     j = 0
        #     for _ in range(self.n_parallel * 2 + 2):
        #         if step_B_blocker.acquire(block=False):
        #             j += 1
        #         else:
        #             [step_B_blocker.release() for _ in range(j)]
        #             break
        #     print("Acquired Step_B_blocker this many times: ", j)

