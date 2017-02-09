import random
from rllab import spaces

import numpy as np

from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


class GoalGenerator(object):
    """ Base class for goal generator. """

    def __init__(self):
        self._goal = None
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
        random.seed()
        super(UniformListGoalGenerator, self).__init__()

    def update(self):
        self._goal = random.choice(self.goal_list)
        return self.goal


class UniformGoalGenerator(GoalGenerator, Serializable):
    """ Generating goals uniformly from a goal list. """

    def __init__(self, goal_dim, bound=2):
        Serializable.quick_init(self, locals())
        self.goal_dim = goal_dim
        self.bound = bound
        super(UniformGoalGenerator, self).__init__()

    def update(self):
        self._goal = np.random.uniform(low=-self.bound, high=self.bound, size=self.goal_dim)
        # print("new goal: ", self._goal)
        return self.goal


class FixedGoalGenerator(GoalGenerator, Serializable):
    """ Generating a fixed goal. """

    def __init__(self, goal):
        Serializable.quick_init(self, locals())
        super(FixedGoalGenerator, self).__init__()
        self._goal = goal


class GoalEnv(Serializable):
    """ Base class for goal based environment. Implements goal update utilities. """

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


class GoalExplorationEnv(GoalEnv, ProxyEnv, Serializable):
    def __init__(self, env, goal_generator, goal_weight, distance_metric='L2',
                 goal_reward='NegativeDistance', inner_weight=0):

        Serializable.quick_init(self, locals())
        self.update_goal_generator(goal_generator)
        self._distance_metric = distance_metric
        self._goal_reward = goal_reward
        self.goal_weight = goal_weight
        self.inner_weight = inner_weight
        ProxyEnv.__init__(self, env)

    def reset(self, reset_goal=True, reset_inner=True):
        if reset_goal:
            self.update_goal()
        if reset_inner:
            return self._append_observation(ProxyEnv.reset(self))
        return self.get_current_obs()

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        body_com = observation[-3:-1]  # assumes the COM is last 3 coord, z being last
        info['distance'] = np.linalg.norm(body_com - self.current_goal)
        reward_dist = self._compute_dist_reward(body_com)
        info['reward_dist'] = reward_dist
        return (
            self._append_observation(observation),
            reward_dist + reward_inner,
            done,
            info
        )

    def _compute_dist_reward(self, obs):
        if self._distance_metric == 'L1':
            goal_distance = np.sum(np.abs(obs - self.current_goal))
        elif self._distance_metric == 'L2':
            goal_distance = np.sum(np.square(obs - self.current_goal))
        elif callable(self._distance_metric):
            goal_distance = self._distance_metric(obs, self.current_goal)
        else:
            raise NotImplementedError('Unsupported distance metric type.')

        if self._goal_reward == 'NegativeDistance':
            intrinsic_reward = - goal_distance
        elif self._goal_reward == 'InverseDistance':
            intrinsic_reward = 1. / (goal_distance + 0.1)
        elif callable(self._goal_reward):
            intrinsic_reward = self._goal_reward(goal_distance)
        else:
            raise NotImplementedError('Unsupported goal_reward type.')

        return self.goal_weight * intrinsic_reward

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
    def log_diagnostics(self, paths, *args, **kwargs):
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
        # Process by trajectories
        logger.record_tabular('InitGoalDistance', np.mean(initial_goal_distances))
        logger.record_tabular('MeanDistance', np.mean(distances))
        logger.record_tabular('MeanRewardDist', np.mean(reward_dist))
        logger.record_tabular('MeanRewardInner', np.mean(reward_inner))


def update_env_goal_generator(env, goal_generator):
    """ Update the goal generator for normalized environment. """
    if hasattr(env, 'update_goal_generator'):
        return env.update_goal_generator(goal_generator)
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.update_goal_generator(goal_generator)
    else:
        raise NotImplementedError('Unsupported environment')
