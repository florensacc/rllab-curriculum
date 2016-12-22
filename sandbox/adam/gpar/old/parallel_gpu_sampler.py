from rllab.misc import tensor_utils
import numpy as np
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
from rllab.sampler.base import BaseSampler
# import copy
import gtimer as gt


class ParallelGpuSampler(BaseSampler):

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
            current_obs=np.ctypeslib.as_array(
                mp.RawArray('f', n_par * obs_dim)).reshape(n_par, obs_dim),
            current_act=np.ctypeslib.as_array(
                mp.RawArray('f', n_par * act_dim)).reshape(n_par, act_dim),
            current_rew=np.ctypeslib.as_array(mp.RawArray('f', n_par)),
            current_done=mp.RawArray(c_bool, n_par),
            continue_sampling=mp.RawValue(c_bool),
        )
        for k in state_info_keys:
            shareds.k = np.ctypeslib.as_array(mp.RawArray('f', n_par))  # Assume scalar for now
        barriers = struct(
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
        range_par = range(self.n_parallel)

        paths = []
        cum_length_complete_paths = 0
        sims_act = [[] for _ in range_par]
        sims_obs = [[] for _ in range_par]
        sims_rew = [[] for _ in range_par]
        sims_agent_info = [[] for _ in range_par]

        # Start-up: do not record yet.
        par.barriers.step_done.wait()
        gt.stamp('bar_first_step')
        all_obs = par.shareds.current_obs.copy()
        all_act, all_agent_info = self.algo.policy.get_actions(all_obs)  # could instead send shareds.current_obs
        par.shareds.current_act[:] = all_act  # copy just for building the paths.
        gt.stamp('first_act')
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

            par.barriers.act_ready.wait()
            # gt.stamp('bar_act')

            # All of the below happens while the simulators are stepping.

            # Move current data into persistent variables.
            # TODO: think about how to copy just into final concatenated arrays?
            # No, can't do that, because...oh actually can do that, just the paths
            # will be interleaved with each other, and so I'll have to keep track
            # of the order for writing the advantages, as well.  Eh maybe this
            # existing way will be fast enough anyway.
            for rank in range_par:
                sims_act[rank].append(all_act[rank, :])  # use new 'all_act' each iteration instead of shared_act, to make sure it's not just a view to shared variable, which will change values
                sims_rew[rank].append(all_rew[rank])
                sims_obs[rank].append(all_obs[rank, :])
                sims_agent_info[rank].append({k: v[rank] for k, v in all_agent_info.items()})
                if all_done[rank]:
                    paths.append(dict(
                        observations=tensor_utils.stack_tensor_list(sims_obs[rank]),
                        actions=tensor_utils.stack_tensor_list(sims_act[rank]),
                        rewards=tensor_utils.stack_tensor_list(sims_rew[rank]),
                        agent_infos=tensor_utils.stack_tensor_dict_list(sims_agent_info[rank]),
                        env_infos={},
                    ))
                    cum_length_complete_paths += len(sims_rew[rank])
                    continue_sampling = (cum_length_complete_paths < self.batch_size)
                    sims_act[rank] = []
                    sims_obs[rank] = []
                    sims_rew[rank] = []
                    sims_agent_info[rank] = []
            all_obs = next_obs
            all_act = next_act
            all_agent_info = next_agent_info

        # loop.exit()
        gt.stamp('samp')
        print("Master exited sampling loop: ", itr, "at count: ", i)

        # Have to go through these one more time for the simulators to get
        # the signal to leave the while loop.
        par.barriers.step_done.wait()
        par.shareds.continue_sampling.value = False
        par.barriers.act_ready.wait()

        return paths

    @gt.wrap
    def obtain_samples_simulator(self, rank, itr):
        print("Rank: ", rank, "entering obtain itr: ", itr)
        par = self.par_objs

        # Start-up: provide first observation.
        par.shareds.current_obs[rank, :] = self.algo.env.reset()

        par.barriers.enter_obtain.wait()
        print("Rank: ", rank, "past barrier enter obtain itr: ", itr)
        par.barriers.step_done.wait()
        gt.stamp('bar_first_step')
        par.barriers.act_ready.wait()
        gt.stamp('bar_first_act')
        print("Rank: ", rank, "starting sampling loop itr: ", itr)

        # loop = gt.timed_loop('samp', save_itrs=False)
        i = 0
        while par.shareds.continue_sampling.value:
            i += 1
            # next(loop)
            # Step the simulator.
            o, r, d, env_info = self.algo.env.step(par.shareds.current_act[rank, :])
            # gt.stamp('step')
            # TODO: Later, might want to have the simulator skip an iteration to reset.
            if d:
                o = self.algo.env.reset()
                # gt.stamp('reset')

            # Write all the current data.
            par.shareds.current_obs[rank, :] = o
            par.shareds.current_rew[rank] = r
            par.shareds.current_done[rank] = d
            # Do nothing with env_info yet.
            # gt.stamp('share')
            # Signal that writing is complete.
            par.barriers.step_done.wait()
            # gt.stamp('bar_step')
            # Wait for master to return action or loop condition.
            # NOTE: evaluate loop condition after this barrier.
            par.barriers.act_ready.wait()
            # gt.stamp('bar_act')
        # loop.exit()
        print("Rank: ", rank, "exited sampling loop: ", itr, "at count: ", i)
        gt.stamp('samp', qp=True)
