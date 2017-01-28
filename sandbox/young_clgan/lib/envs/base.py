import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides


class GoalGenerator(Serializable):
    """ Base class for goal generator. """

    def __init__(self):
        self._goal = None
        self.update()

    def update(self):
        return self.goal

    @property
    def goal(self):
        return self._goal


class UniformGoalGenerator(GoalGenerator):
    """ Generating goals uniformly from a goal list. """

    def __init__(self, goal_list):
        self.goal_list = goal_list
        random.seed()
        Serializable.quick_init(self, locals())
        self.update()

    def update(self):
        self._goal = random.choice(self.goal_list)
        return self.goal


class FixedGoalGenerator(GoalGenerator):
    """ Generating a fixed goal. """

    def __init__(self, goal):
        self._goal = goal
        Serializable.quick_init(self, locals())


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
            goal_reward='Negative Distance'):

        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.update_goal_generator(goal_generator)
        self._distance_metric = distance_metric
        self._goal_reward = goal_reward
        self.goal_weights = goal_weights

    def reset(self):
        self.update_goal()
        return self._append_observation(ProxyEnv.reset(self))

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        return (
            self._append_observation(observation),
            self._compute_reward(reward),
            done,
            info
        )

    def _append_observation(self, obs):
        return np.concatentate(obs, np.array(self.current_goal))

    def _compute_reward(self, obs, reward):
        if self._distance_metric == 'L1':
            goal_distance = np.sum(np.abs(obs - self.current_goal))
        elif self._distance_metric == 'L2':
            goal_distance = np.sum(np.square(obs - self.current_goal))
        elif callable(self._distance_metric):
            goal_distance = self._distance_metric(obs, self.current_goal)
        else:
            raise NotImplementedError('Unsupported distance metric type.')

        if self._goal_reward == 'Negative Distance':
            intrinsic_reward = - goal_distance
        elif callable(self._goal_reward):
            intrinsic_reward = self._goal_reward(goal_distance)
        else:
            raise NotImplementedError('Unsupported goal_reward type.')

        return goal_weight * intrinsic_reward + reward


def update_env_goal_generator(env, goal_generator):
    """ Update the goal generator for normalized environment. """
    if hasattr(env, 'update_goal_generator'):
        return env.update_goal_generator(goal_generator)
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.update_goal_generator(goal_generator)
    else:
        raise NotImplementedError('Unsupported environment')
