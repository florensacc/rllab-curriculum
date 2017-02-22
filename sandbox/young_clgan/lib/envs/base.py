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
from sandbox.young_clgan.lib.envs.rewards import linear_threshold_reward


class GoalGenerator(object):
    """ Base class for goal generator. """

    def __init__(self):
        self._goal = None
        # self.goal_dim = 0
        self.update()

    def update(self):
        return self.goal

    @property
    def goal(self):
        return self._goal


class UniformListGoalGenerator(GoalGenerator, Serializable):
    """ Generating goals uniformly from a goal list. """

    def __init__(self, goal_list):
        Serializable.quick_init(self, locals())
        self.goal_list = goal_list
        self.goal_size = np.size(self.goal_list[0])  # assumes all goals have same dim as first in list
        random.seed()
        super(UniformListGoalGenerator, self).__init__()

    def update(self):
        self._goal = random.choice(self.goal_list)
        return self.goal


class UniformGoalGenerator(GoalGenerator, Serializable):
    """ Generating goals uniformly from a goal list. """

    def __init__(self, goal_size, bounds=2, center=()):
        Serializable.quick_init(self, locals())
        self.goal_size = goal_size
        self.bounds = bounds
        if np.array(self.bounds).size == 1:
            self.bounds = [-1 * bounds * np.ones(goal_size), bounds * np.ones(goal_size)]
        print(self.bounds)
        self.center = center if len(center) else np.zeros(self.goal_size)
        super(UniformGoalGenerator, self).__init__()

    def update(self):  # This should be centered around the initial position!!
        sample = []
        for low, high in zip(*self.bounds):
            sample.append(np.random.uniform(low, high))
        self._goal = self.center + np.array(sample)
        return self.goal


class FixedGoalGenerator(GoalGenerator, Serializable):
    """ Generating a fixed goal. """

    def __init__(self, goal):
        Serializable.quick_init(self, locals())
        super(FixedGoalGenerator, self).__init__()
        self._goal = goal


class GoalEnv(Serializable):
    """ Base class for goal based environment. Implements goal update utilities. """
    def __init__(self, goal_bounds=None):
        """:param goal_bounds: scalar if square bounds or array if rectangle"""
        self.goal_bounds = goal_bounds

    def update_goal_generator(self, goal_generator):
        self._goal_generator = goal_generator

    def update_goal(self):
        return self.goal_generator.update()

    @property
    def goal_generator(self):
        return self._goal_generator

    @property
    def current_goal(self):
        return self.goal_generator.goal

    def __getstate__(self):
        d = super(GoalEnv, self).__getstate__()
        d['__goal_generator'] = self.goal_generator
        return d

    def __setstate__(self, d):
        super(GoalEnv, self).__setstate__(d)
        self.update_goal_generator(d['__goal_generator'])


class GoalEnvAngle(GoalEnv, Serializable):

    def __init__(self, angle_idxs=(None,), **kwargs):
        """Indicates the coordinates that are angles and need to be duplicated to cos/sin"""
        Serializable.quick_init(self, locals())
        self.angle_idxs = angle_idxs
        GoalEnv.__init__(self, **kwargs)

    @overrides
    @property
    def current_goal(self):
        # print("the goal generator is:", self.goal_generator)
        angle_goal = self.goal_generator.goal
        full_goal = []
        for i, coord in enumerate(angle_goal):
            if i in self.angle_idxs:
                full_goal.extend([np.sin(coord), np.cos(coord)])
            else:
                full_goal.append(coord)
            # print("the angle goal is: {}, the full goal is: {}".format(angle_goal, full_goal))
        return full_goal


class GoalExplorationEnv(GoalEnvAngle, ProxyEnv, Serializable):
    def __init__(self, env, goal_generator, terminal_bonus=0, terminal_eps=0.1, reward_dist_threshold=None, final_goal=None,
                 distance_metric='L2', goal_reward='NegativeDistance', goal_weight=1,
                 inner_weight=0, angle_idxs=(None,), **kwargs):
        """
        :param env: wrapped env
        :param goal_generator: already instantiated: NEEDS GOOD DIM OF GOALS! --> TO DO: give the class a method to update dim?
        :param terminal_bonus: if not 0, the rollout terminates with this bonus if the goal state is reached
        :param terminal_eps: eps around which the terminal goal is considered reached
        :param distance_metric: L1 or L2 or a callable func
        :param goal_reward: NegativeDistance or InverseDistance or callable func
        :param goal_weight: coef of the goal-dist reward
        :param inner_weight: coef of the inner reward
        """

        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.update_goal_generator(goal_generator)
        self.terminal_bonus = terminal_bonus
        self.terminal_eps = terminal_eps
        self.reward_dist_threshold = reward_dist_threshold
        self._distance_metric = distance_metric
        self._goal_reward = goal_reward
        self.goal_weight = goal_weight
        self.inner_weight = inner_weight
        self.fig_number = 0
        self.final_goal = final_goal
        GoalEnvAngle.__init__(self, angle_idxs=angle_idxs, **kwargs)
        if self.goal_bounds is None:
            self.goal_bounds = self.wrapped_env.observation_space.bounds
        # elif np.array(self.goal_bounds).size <= 1:
        #     self.goal_bounds = [-1 * self.goal_bounds * np.ones(self.wrapped_env.observation_space.flat_dim),
        #                         self.goal_bounds * np.ones(self.wrapped_env.observation_space.flat_dim)]

    def reset(self, fix_goal=None, reset_goal=True, reset_inner=True):
        if reset_goal:
            self.update_goal()
            if fix_goal is not None:
                self.goal_generator._goal = fix_goal
        print("RESET goal to:", self.goal_generator.goal)
        if reset_inner:
            return self._append_observation(ProxyEnv.reset(self))
        return self.get_current_obs()

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        info['distance'] = dist = self._compute_dist(observation)
        info['reward_dist'] = reward_dist = self._compute_dist_reward(observation)
        if self.terminal_bonus and dist <= self.terminal_eps:
            print("*****done!!*******")
            done = True
            reward_dist += self.terminal_bonus
        return (
            self._append_observation(observation),
            reward_dist + reward_inner,
            done,
            info
        )

    def _compute_dist_reward(self, obs):
        goal_distance = self._compute_dist(obs)
        if self.reward_dist_threshold is not None:
            intrinsic_reward = linear_threshold_reward(goal_distance, threshold=self.reward_dist_threshold,
                                                       coefficient=-1000)  # this should also be a hyper!!
        elif self._goal_reward == 'NegativeDistance':
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
            goal_distance = np.sqrt(np.sum(np.square(obs - self.current_goal)))
        elif callable(self._distance_metric):
            goal_distance = self._distance_metric(obs, self.current_goal)
        else:
            raise NotImplementedError('Unsupported distance metric type.')
        return goal_distance

    def get_current_obs(self):
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env
        return self._append_observation(obj.get_current_obs())

    def _append_observation(self, obs):
        return np.concatenate([obs, np.array(self.current_goal)])

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @overrides
    def log_diagnostics(self, paths, fig_prefix='', *args, **kwargs):
        if fig_prefix == '':
            fig_prefix = str(self.fig_number)
            self.fig_number +=1
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
        print(success)

        # compute also the distance to the ultimate goal we care about


        # Can I also log the goal_success rate??

        # Process by trajectories
        logger.record_tabular('InitGoalDistance', np.mean(initial_goal_distances))
        logger.record_tabular('MeanDistance', np.mean(distances))
        logger.record_tabular('MeanRewardDist', np.mean(reward_dist))
        logger.record_tabular('MeanRewardInner', np.mean(reward_inner))
        logger.record_tabular('SuccessRate', np.mean(success))

        # # The goal itself is prepended to the observation, so we can retrieve the collection of goals:
        # full_goal_dim = np.size(self.current_goal)
        # goals = [path['observations'][0][-full_goal_dim:] for path in paths]  # supposes static goal over whole paths
        # angle_goals = [math.atan2(goal[0], goal[1]) for goal in goals]
        # angVel_goals = [goal[2] for goal in goals]
        # colors = ['g'*succ + 'r'*(1-succ) for succ in success]
        # fig, ax = plt.subplots()
        # ax.scatter(angle_goals, angVel_goals, c=colors, lw=0)

        # log_dir = logger.get_snapshot_dir()
        # plt.savefig(osp.join(log_dir, fig_prefix + 'goal_performance.png'))
        # plt.close()



class GoalIdxExplorationEnv(GoalExplorationEnv, Serializable):
    """
    Instead of using the full state-space as goal, this class uses only some idx of observation ([-3,-1] CoM in MuJoCo)
    """

    def __init__(self, idx=(-3, -2), **kwargs):
        Serializable.quick_init(self, locals())
        self.idx = idx
        super(GoalIdxExplorationEnv, self).__init__(**kwargs)

    def step(self, action):
        # print("action: ", action)
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        body_com = observation[self.idx,]  # assumes the COM is last 3 coord, z being last
        info['distance'] = dist = np.linalg.norm(body_com - self.current_goal)
        reward_dist = self._compute_dist_reward(body_com)
        info['reward_dist'] = reward_dist
        if self.terminal_bonus and dist <= self.terminal_eps:
            print("*****done!!*******")
            done = True
            reward_dist += self.terminal_bonus
        return (
            self._append_observation(observation),
            reward_dist + reward_inner,
            done,
            info
        )

def update_env_goal_generator(env, goal_generator):
    """ Update the goal generator for normalized environment. """
    if hasattr(env, 'update_goal_generator'):
        return env.update_goal_generator(goal_generator)
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.update_goal_generator(goal_generator)
    else:
        raise NotImplementedError('Unsupported environment')
