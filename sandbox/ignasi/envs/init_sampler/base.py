import random
from rllab import spaces
import sys
import os.path as osp
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import math

from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.young_clgan.state.selectors import FixedStateSelector


class InitBase(Serializable):
    """ Base class for init based environment. Implements init update utilities. """

    def __init__(self, init_selector, goal_selector):
        """
        :param init_selector: generator of stateial positions
        :param goal: point in state-space that gives a reward
        """
        Serializable.quick_init(self, locals())
        self._init_selector = init_selector
        self._goal_selector = goal_selector

    @property
    def current_init(self):  # this will later be of the dim needed to reset the inner env
        return self.init_selector.state

    @property
    def current_goal(self):  # this will always be of the output dim of the selector
        return self.goal_selector.state

    @property
    def init_selector(self):
        return self._init_selector

    @property
    def goal_selector(self):
        return self._goal_selector

    def update_init_selector(self, init_selector):
        self._init_selector = init_selector

    def update_goal_selector(self, goal_selector):
        self._goal_selector = goal_selector

    def update_init(self):
        return self.init_selector.update()

    def update_goal(self):
        return self.goal_selector.update()

    def convert2selector(self, full_state):
        return full_state

    def __getstate__(self):
        d = super(InitBase, self).__getstate__()
        d['__init_selector'] = self.init_selector
        return d

    def __setstate__(self, d):
        super(InitBase, self).__setstate__(d)
        self.update_init_selector(d['__init_selector'])


class InitBaseAngle(InitBase, Serializable):
    """
    The selectors have one dim per angle, but the env has 2 per angle!
    In general, current_init/goal will have the dim of the state-space, which might be different from the selectors!
    """

    def __init__(self, angle_idxs=(None,), **kwargs):
        """Indicates the coordinates that are angles and need to be duplicated to cos/sin"""
        Serializable.quick_init(self, locals())
        self.angle_idxs = angle_idxs
        super(InitBaseAngle, self).__init__(**kwargs)

    @overrides
    @property
    def current_init(self):
        angle_init = self.init_selector.state
        full_init = []
        for i, coord in enumerate(angle_init):
            if i in self.angle_idxs:
                full_init.extend([np.sin(coord), np.cos(coord)])
            else:
                full_init.append(coord)
        return full_init

    @overrides
    @property
    def current_goal(self):
        angle_goal = self.goal_selector.state
        full_goal = []
        for i, coord in enumerate(angle_goal):
            if i in self.angle_idxs:
                full_goal.extend([np.sin(coord), np.cos(coord)])
            else:
                full_goal.append(coord)
        return full_goal

    @overrides
    def convert2selector(self, full_state):  # used for feasible init
        i = 0
        selector_state = []
        while i < len(full_state):
            if i in self.angle_idxs:
                ang = np.arccos(full_state[i + 1])
                if full_state[i] < 0:
                    ang = - ang
                selector_state.append(ang)
                i += 2
            else:
                selector_state.append(full_state[i])
                i += 1
        return selector_state


class InitEnv(InitBaseAngle, ProxyEnv, Serializable):
    def __init__(self, env, init_selector, goal_selector, append_goal=False, terminal_bonus=0, terminal_eps=0.1,
                 distance_metric='L2', goal_reward='NegativeDistance', goal_weight=1,
                 inner_weight=0, angle_idxs=(None,), persistence=3):
        """
        :param env: wrapped env  NEEDS RESET WITH INIT_STATE arg!
        :param init_selector: already instantiated: NEEDS GOOD DIM OF GOALS! --> TO DO: give the class a method to update dim?
        :param terminal_bonus: if not 0, the rollout terminates with this bonus if the goal state is reached
        :param terminal_eps: eps around which the ter/minal goal is considered reached
        :param distance_metric: L1 or L2 or a callable func
        :param goal_reward: NegativeDistance or InverseDistance or callable func
        :param goal_weight: coef of the goal-dist reward
        :param inner_weight: coef of the inner reward
        """

        Serializable.quick_init(self, locals())
        self.append_goal = append_goal
        self.terminal_bonus = terminal_bonus
        self.terminal_eps = terminal_eps
        self._distance_metric = distance_metric
        self._goal_reward = goal_reward
        self.goal_weight = goal_weight
        self.inner_weight = inner_weight
        self.persistence = persistence
        self.persistence_count = 1
        self.fig_number = 0
        ProxyEnv.__init__(self, env)
        InitBaseAngle.__init__(self, angle_idxs=angle_idxs, init_selector=init_selector, goal_selector=goal_selector)
        ub = BIG * np.ones(self.get_current_obs().shape)
        self._observation_space = spaces.Box(ub * -1, ub)
        # keep around all sampled goals, one dict per training itr (ie, call of this env.log_diagnostics)
        self.inits_trained = []

    @overrides
    def reset(self, force_reset_init=False, reset_inner=True):
        if force_reset_init or self.persistence_count >= self.persistence:
            self.update_init()
            self.persistence_count = 1
        else:
            self.persistence_count += 1
        if reset_inner:
            self.wrapped_env.reset(init_state=self.current_init)
        obs = self.get_current_obs()
        return obs

    @overrides
    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        info['distance'] = dist = self._compute_dist(observation)
        info['reward_dist'] = reward_dist = self._compute_dist_reward(observation)
        if self.terminal_bonus and dist <= self.terminal_eps:
            # print("*****done!!*******")
            done = True
            reward_dist += self.terminal_bonus
        return (
            self.get_current_obs(),
            reward_dist + reward_inner,
            done,
            info
        )

    def _compute_dist_reward(self, obs):
        goal_distance = self._compute_dist(obs)
        if self._goal_reward == 'NegativeDistance':
            intrinsic_reward = - goal_distance
        elif self._goal_reward == 'InverseDistance':
            intrinsic_reward = 1. / (goal_distance + 0.1)
        elif callable(self._goal_reward):
            intrinsic_reward = self._goal_reward(goal_distance)
        else:
            raise NotImplementedError('Unsupported goal_reward type.')

        return self.goal_weight * intrinsic_reward

    def _compute_dist(self, obs):
        if self._distance_metric == 'L1':
            goal_distance = np.sum(np.abs(obs - self.current_goal))
        elif self._distance_metric == 'L2':
            goal_distance = np.sqrt(np.sum(np.square(obs[:np.size(self.current_goal)] - self.current_goal)))
        elif callable(self._distance_metric):
            goal_distance = self._distance_metric(obs, self.current_goal)
        else:
            raise NotImplementedError('Unsupported distance metric type.')
        # import pdb; pdb.set_trace()
        # print(obs, self.current_goal, '--> goal_dist: ', goal_distance)
        return goal_distance

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    @overrides
    def get_current_obs(self):
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env
        if self.append_goal:
            return np.concatenate([obj.get_current_obs(), np.array(self.current_goal)])
        else:
            return obj.get_current_obs()

    @overrides
    def convert2selector(self, full_state):
        if self.append_goal:
            full_state = full_state[:-np.size(self.current_goal)]
        super(self, InitEnv).convert2selector(full_state)

    @overrides
    def log_diagnostics(self, paths, fig_prefix='', *args, **kwargs):
        if fig_prefix == '':
            fig_prefix = str(self.fig_number)
            self.fig_number += 1
        # Process by time steps
        distances = [
            np.mean(path['env_infos']['distance'])
            for path in paths
        ]
        initial_goal_distances = [
            path['env_infos']['distance'][0] for path in paths
        ]
        reward_dist = [
            np.mean(path['env_infos']['reward_dist'])
            for path in paths
        ]
        reward_inner = [
            np.mean(path['env_infos']['reward_inner'])
            for path in paths
        ]
        success = [int(np.min(path['env_infos']['distance']) <= self.terminal_eps) for path in paths]
        # print(success)

        # Process by trajectories
        logger.record_tabular('InitGoalDistance', np.mean(initial_goal_distances))
        logger.record_tabular('MeanDistance', np.mean(distances))
        logger.record_tabular('MeanRewardDist', np.mean(reward_dist))
        logger.record_tabular('MeanRewardInner', np.mean(reward_inner))
        logger.record_tabular('SuccessRate', np.mean(success))

        # build a dict of the init states that were actually trained on
        inits_used = {}
        for path in paths:
            init = tuple(self.convert2selector(path['observations'][0]).reshape((-1,)))  # here observation is with the appended goal.
            if init not in inits_used.keys():
                inits_used[init] = [np.sum(path['rewards'])]
            else:
                inits_used[init].append(np.sum(path['rewards']))
        # convert to numerical label
        for key in inits_used.keys():
            inits_used[key] = np.mean(inits_used[key])
        self.inits_trained.append(inits_used)


        # colors = ['g' * succ + 'r' * (1 - succ) for succ in success]
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # # The goal itself is appended to the observation, so we can retrieve the collection of goals:
        # full_goal_dim = np.size(self.current_goal)
        # if self.append_goal:
        #     inits = [path['observations'][0][:-full_goal_dim] for path in
        #              paths]  # supposes static goal over whole paths
        # else:
        #     inits = [path['observations'][0] for path in paths]  # supposes static goal over whole paths
        # # angle_inits = [math.atan2(init[0], init[1]) for init in inits]
        # # angVel_inits = [init[2] for init in inits]
        # # ax.scatter(angle_inits, angVel_inits, c=colors, lw=0)
        # init_x = [init[0] for init in inits]
        # init_y = [init[1] for init in inits]
        # ax.scatter(init_x, init_y, c=colors, lw=0)
        # ax.scatter(self.current_goal[0], self.current_goal[1], c='b')
        # log_dir = logger.get_snapshot_dir()
        # print("trying to save the figure at: ", osp.join(log_dir, fig_prefix + 'init_performance.png'))
        # fig.savefig(osp.join(log_dir, fig_prefix + 'init_performance.png'))
        # plt.close()


class InitIdxEnv(InitEnv, Serializable):
    """
    Instead of using the full state-space as goal, this class uses the observation[-3,-1] CoM in MuJoCo
    """

    def __init__(self, idx=(-3, -2), **kwargs):
        """:param idx: set of entries to ignore in the state when comparing to current_goal"""
        Serializable.quick_init(self, locals())
        self.idx = idx
        super(InitIdxEnv, self).__init__(**kwargs)

    @overrides
    @property
    def current_init(self):  # ready to use in reset
        ang_expanded_init = super(InitIdxEnv, self).current_init
        inner_obs_dim = self.wrapped_env.observation_space.flat_dim
        init = np.zeros(inner_obs_dim)
        init[self.idx,] = ang_expanded_init
        return init

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        obsIdx = observation[self.idx,]
        info['distance'] = dist = np.linalg.norm(obsIdx - self.current_goal)
        info['reward_dist'] = reward_dist = self._compute_dist_reward(obsIdx)
        if self.terminal_bonus and dist <= self.terminal_eps:
            # print("*****done!!*******")
            done = True
            reward_dist += self.terminal_bonus
        return (
            self.get_current_obs(),
            reward_dist + reward_inner,
            done,
            info
        )

    @overrides
    def convert2selector(self, full_state):
        return full_state[self.idx, ]  # this is also "robuts" to the appending of the goal


# def update_env_init_selector(env, init_selector):
#     """ Update the init selector for normalized environment. """
#     obj = env
#     while not hasattr(obj, 'update_init_selector') and hasattr(obj, 'wrapped_env'):
#         obj = obj.wrapped_env
#     if hasattr(obj, 'update_init_selector'):
#         return obj.update_init_selector(init_selector)
#     else:
#         raise NotImplementedError('Unsupported environment')
#
#
# def update_env_goal_selector(env, goal_selector):
#     """ Update the goal selector for normalized environment. """
#     obj = env
#     while not hasattr(obj, 'update_init_selector') and hasattr(obj, 'wrapped_env'):
#         obj = obj.wrapped_env
#     if hasattr(obj, 'update_init_selector'):
#         return obj.update_goal_selector(goal_selector)
#     else:
#         raise NotImplementedError('Unsupported environment')
#
#
# def get_selector_observation(env):
#     """ Extract selector formatted from environment state. """
#     obj = env
#     while not hasattr(obj, 'convert2selector') and hasattr(obj, 'wrapped_env'):
#         obj = obj.wrapped_env
#     if hasattr(obj, 'convert2selector'):
#         return obj.convert2selector(obj.wrapped_env.get_current_obs())
#     else:
#         raise NotImplementedError('Unsupported environment')
#
#
# def get_current_goal(env):
#     if hasattr(env, 'current_goal'):
#         return env.current_goal
#     elif hasattr(env, 'wrapped_env'):
#         return env.wrapped_env.current_goal
#     else:
#         raise NotImplementedError('Unsupported environment')


def generate_initial_inits(env, policy, max_path_length=500, size=10000):
    obs = env.reset()
    inits = [env.convert2selector(env.wrapped_env.get_current_obs())]
    steps = 0
    while len(inits) < size:
        steps += 1
        if steps >= max_path_length:  # this is bad for unstable inner envs: won't kill rollout when bad!
            steps = 0
            env.update_init_selector(FixedStateSelector(state=env.current_goal))
            obs = env.reset()
            inits.append(env.convert2selector(env.wrapped_env.get_current_obs()))
        else:
            action, _ = policy.get_action(obs)
            obs, _, done, _ = env.step(action)
            inits.append(env.convert2selector(env.wrapped_env.get_current_obs()))
    return np.array(inits)
