from rllab.misc import tensor_utils
import numpy as np
import multiprocessing as mp
from sandbox.adam.util import struct
from ctypes import c_bool
from rllab.sampler.base import BaseSampler
from rllab.sampler.utils import rollout
from rllab.misc import logger
import pyprind
from rllab.algos import util
from rllab.misc import special
# import copy
import gtimer as gt
import time


class PairwiseGpuSampler_4(BaseSampler):

    def __init__(self, algo):
        """
        :type algo: ParallelGpuBatchPolopt
        """
        self.algo = algo
        self.n_parallel = self.algo.n_parallel
        self.n_simulators = self.algo.n_simulators
        self.batch_size = self.algo.batch_size

        obs_dim = int(self.algo.env.observation_space.flat_dim)
        act_dim = int(self.algo.env.action_space.flat_dim)
        self.init_par_objs(self.n_parallel, self.n_simulators,
            obs_dim, act_dim)  # (before processes fork)

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def obtain_example_samples(self, path_length=10, num_paths=2):
        paths = []
        for _ in range(num_paths):
            paths.append(
                rollout(self.algo.env, self.algo.policy, max_path_length=path_length))
        self.algo.env.reset()
        self.algo.policy.reset()
        return paths

    def process_example_samples(self, paths):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)

        return samples_data

    def init_par_objs(self, n_par, n_sim, obs_dim, act_dim):
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
            # act=[np.ctypeslib.as_array(mp.RawArray('i', n)) for _ in range(2)],
            rew=[np.ctypeslib.as_array(mp.RawArray('f', n)) for _ in range(2)],
            done=[mp.RawArray(c_bool, n) for _ in range(2)],
            continue_sampling=mp.RawValue(c_bool, True),
        )
        semaphores = struct(
            # step_waiter=(mp.Semaphore(0), mp.Semaphore(0)),
            # step_blocker=(mp.Semaphore(n_par - 1), mp.Semaphore(n_par - 1)),
            step_blockers=([mp.Semaphore(0) for _ in range(n_par)],
                           [mp.Semaphore(0) for _ in range(n_par)]),
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
        # step_waiter = par.semaphores.step_waiter
        step_blockers = par.semaphores.step_blockers

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

        # step_waiter[0].acquire()  # gates.step[0].wait()
        for blocker in step_blockers[0]:
            blocker.acquire()

        # Start-up: do not record yet.
        all_obs[0] = par.shareds.obs[0].copy()
        act, all_agent_info[0] = self.algo.policy.get_actions(all_obs[0])  # could instead send shareds.current_obs
        all_act[0] = self.algo.env.action_space.flatten_n(act)
        par.shareds.act[0][:] = all_act[0]  # copy just for building the paths.

        [w.release() for w in act_waiters[0]]  # gates.act[0].open()

        # step_waiter[1].acquire()  # gates.step[1].wait()
        for blocker in step_blockers[1]:
            blocker.acquire()

        all_obs[1] = par.shareds.obs[1].copy()
        act, all_agent_info[1] = self.algo.policy.get_actions(all_obs[1])
        all_act[1] = self.algo.env.action_space.flatten_n(act)
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

            # step_waiter[j].acquire()  # gates.step[j].wait()
            for blocker in step_blockers[j]:
                blocker.acquire()

            all_rew[j] = par.shareds.rew[j].copy()
            next_obs[j] = par.shareds.obs[j].copy()
            all_done[j] = par.shareds.done[j][:]

            # gt.stamp('copy')
            act, next_agent_info[j] = self.algo.policy.get_actions(next_obs[j])
            par.shareds.act[j][:] = self.algo.env.action_space.flatten_n(act)
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
                        env_infos={},  # (have yet to see an env provide this)
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
        # step_waiter[j].acquire()  # gates.step[j].wait()
        for blocker in step_blockers[j]:
            blocker.acquire()

        # Now the simulators are waiting at the action of (j ^ 1)
        par.shareds.continue_sampling.value = False
        j = j ^ 1
        [w.release() for w in act_waiters[j]]  # gates.act[j].open()
        # Now the simulators check the while condition and exit.
        # Just to close the gate, (simulators opened it):
        # step_waiter[j].acquire()  # gates.step[j].wait()
        for blocker in step_blockers[j]:
            blocker.acquire()
        pbar.stop()

        return paths

    @gt.wrap
    def obtain_samples_simulator(self, rank, itr):
        par = self.par_objs

        # step_blocker = par.semaphores.step_blocker
        step_blockers = (par.semaphores.step_blockers[0][rank],
                         par.semaphores.step_blockers[1][rank])
        # step_waiter = par.semaphores.step_waiter
        act_waiter = (par.semaphores.act_waiters[0][rank],
                      par.semaphores.act_waiters[1][rank])
        # release_range = range(self.n_parallel - 1)

        # Assign a block of rows to each rank.
        start_idx = rank * self.n_simulators
        idx = [start_idx + s for s in range(self.n_simulators)]

        # Start-up: provide first observations.
        for s, env in enumerate(self.algo.envs[0]):
            par.shareds.obs[0][idx[s], :] = env.observation_space.flatten(env.reset())
        # par.shareds.obs[0][rank, :] = self.algo.env[0].reset()
        # if not step_blocker[0].acquire(block=False):  # gates.step[0].checkin()
        #     step_waiter[0].release()
        #     [step_blocker[0].release() for _ in release_range]
        step_blockers[0].release()

        for s, env in enumerate(self.algo.envs[1]):
            par.shareds.obs[1][idx[s], :] = env.observation_space.flatten(env.reset())
        # par.shareds.obs[1][rank, :] = self.algo.env[1].reset()
        # if not step_blocker[1].acquire(block=False):  # gates.step[1].checkin()
        #     step_waiter[1].release()
        #     [step_blocker[1].release() for _ in release_range]
        step_blockers[1].release()

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
                a = env.action_space.unflatten(par.shareds.act[j][idx[s], :])
                o, r, d, env_info = env.step(a)
                # gt.stamp('step')
                # TODO: Later, might want to have the simulator skip an iteration to reset.
                if d:
                    o = env.reset()
                    # gt.stamp('reset')

                # Share all the current data.
                par.shareds.obs[j][idx[s], :] = env.observation_space.flatten(o)
                par.shareds.rew[j][idx[s]] = r
                par.shareds.done[j][idx[s]] = d
                # (have yet to see an environment provide env_info)

            # if not step_blocker[j].acquire(block=False):  # gates.step[j].checkin()
            #     step_waiter[j].release()
            #     [step_blocker[j].release() for _ in release_range]
            step_blockers[j].release()
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
