import random
from rllab import spaces
import sys
import os.path as osp

import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
import tempfile
import math

from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from sandbox.young_clgan.envs.rewards import linear_threshold_reward


class StateGenerator(object):
    """ Base class for goal generator. """

    def __init__(self):
        self._state = None
        self.update()

    def update(self):
        return self.state

    @property
    def state(self):
        return self._state


class UniformListStateGenerator(StateGenerator, Serializable):
    """ Generating goals uniformly from a goal list. """

    def __init__(self, state_list):
        Serializable.quick_init(self, locals())
        self.state_list = state_list
        self.state_size = np.size(self.state_list[0])  # assumes all goals have same dim as first in list
        random.seed()
        super(UniformListStateGenerator, self).__init__()

    def update(self):
        self._state = random.choice(self.state_list)
        return self.state


class UniformStateGenerator(StateGenerator, Serializable):
    """ Generating goals uniformly from a goal list. """

    def __init__(self, state_size, bounds=2, center=()):
        Serializable.quick_init(self, locals())
        self.state_size = state_size
        self.bounds = bounds
        if np.array(self.bounds).size == 1:
            self.bounds = [-1 * bounds * np.ones(state_size), bounds * np.ones(state_size)]
        self.center = center if len(center) else np.zeros(self.state_size)
        super(UniformStateGenerator, self).__init__()

    def update(self):  # This should be centered around the initial position!!
        sample = []
        for low, high in zip(*self.bounds):
            sample.append(np.random.uniform(low, high))
        self._state = self.center + np.array(sample)
        return self.state


class FixedStateGenerator(StateGenerator, Serializable):
    """ Generating a fixed goal. """

    def __init__(self, state):
        Serializable.quick_init(self, locals())
        super(FixedStateGenerator, self).__init__()
        self._state = state


class StateAuxiliaryEnv(Serializable):
    """ Base class for state auxiliary environment. Implements state update utilities. """

    def __init__(self, **kwargs):
        Serializable.quick_init(self, locals())

    def update_state_generator(self, state_generator):
        self._state_generator = state_generator

    def update_aux_state(self):
        return self.state_generator.update()

    @property
    def state_generator(self):
        return self._state_generator

    @property
    def current_aux_state(self):
        return self.state_generator.state

    def __getstate__(self):
        d = super(StateAuxiliaryEnv, self).__getstate__()
        d['__state_generator'] = self.state_generator
        return d

    def __setstate__(self, d):
        super(StateAuxiliaryEnv, self).__setstate__(d)
        self.update_state_generator(d['__state_generator'])
        
        
        
        
class GoalEnv(StateAuxiliaryEnv):
    """ A wrapper of StateAuxiliaryEnv to make it compatible with the old goal env."""
    
    def update_goal_generator(self, *args, **kwargs):
        return self.update_state_generator(*args, **kwargs)
        
    def update_goal(self, *args, **kwargs):
        return self.update_aux_state(*args, **kwargs)
        
    @property
    def goal_generator(self):
        return self.state_generator
    
    @property
    def current_goal(self):
        return self.current_aux_state



class GoalExplorationEnv(GoalEnv, ProxyEnv, Serializable):
    def __init__(self, env, goal_generator, obs_transform=None, dist_threshold=0.05,
                 terminate_env=False, goal_bounds=None, distance_metric='L2', goal_weight=1,
                 inner_weight=1, append_transformed_obs=False, **kwargs):
        """
        This environment wraps around a normal environment to facilitate goal based exploration.
        Initial position based experiments should not use this class.
        
        :param env: wrapped env
        :param goal_generator: a StateGenerator object
        :param obs_transform: a callable that transforms an observation of the wrapped environment into goal space
        :param dist_threshold: a threshold of distance that determines if a goal is reached
        :param terminate_env: a boolean that controls if the environment is terminated with the goal is reached
        :param goal_bounds: array marking the UB of the rectangular limit of goals.
        :param distance_metric: L1 or L2 or a callable func
        :param goal_weight: coef of the goal based reward
        :param inner_weight: coef of the inner environment reward
        :param append_transformed_obs: append the transformation of the current observation to full observation
        """

        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.update_goal_generator(goal_generator)
        
        if obs_transform is None:
            self._obs_transform = lambda x: x
        else:
            self._obs_transform = obs_transform
        
        self.terminate_env = terminate_env
        self.goal_bounds = goal_bounds
        self.dist_threshold = dist_threshold
        
        
        self.distance_metric = distance_metric
        self.goal_weight = goal_weight
        self.inner_weight = inner_weight
        
        self.append_transformed_obs = append_transformed_obs

        
        # TODO fix this
        if self.goal_bounds is None:
            # print("setting goal bounds to match env")
            self.goal_bounds = self.wrapped_env.observation_space.bounds[1]  # we keep only UB
            self._feasible_goal_space = self.wrapped_env.observation_space
        else:
            self._feasible_goal_space = Box(low=-1 * self.goal_bounds, high=self.goal_bounds)
            # elif np.array(self.goal_bounds).size <= 1:
            #     self.goal_bounds = [-1 * self.goal_bounds * np.ones(self.wrapped_env.observation_space.flat_dim),
            #                         self.goal_bounds * np.ones(self.wrapped_env.observation_space.flat_dim)]


    @property
    @overrides
    def feasible_goal_space(self):
        return self._feasible_goal_space

    def reset(self, reset_goal=True):
        if reset_goal:
            self.update_goal()
            
        return self._append_goal_observation(ProxyEnv.reset(self))

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        info['distance'] = dist = self._dist_to_goal(observation)
        info['reward_dist'] = reward_dist = self._compute_dist_reward(observation)
        info['reached_goal'] = reward_dist
        if self.terminate_env and dist < self.dist_threshold:
            done = True
        return (
            self._append_goal_observation(observation),
            reward_dist + reward_inner,
            done,
            info
        )

    def _compute_dist_reward(self, observation):
        """ Compute the 0 or 1 reward for reaching the goal. """
        goal_distance = self._dist_to_goal(observation)
        intrinsic_reward = 1.0 * (goal_distance < self.dist_threshold)

        return self.goal_weight * intrinsic_reward

    def _dist_to_goal(self, obs):
        """ Compute the distance of the given observation to the current goal. """
        goal_obs = self.transform_to_goal_space(obs)
        if self.distance_metric == 'L1':
            goal_distance = np.linalg.norm(goal_obs - self.current_goal, ord=1)
        elif self.distance_metric == 'L2':
            goal_distance = np.linalg.norm(goal_obs - self.current_goal, ord=2)
        elif callable(self.distance_metric):
            goal_distance = self.distance_metric(goal_obs, self.current_goal)
        else:
            raise NotImplementedError('Unsupported distance metric type.')
        return goal_distance
        
    def transform_to_goal_space(self, obs):
        """ Apply the goal space transformation to the given observation. """
        return self._obs_transform(obs)    
    
    def get_current_obs(self):
        """ Get the full current observation. The observation should be identical to the one used by policy. """
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env
        return self._append_goal_observation(obj.get_current_obs())

    @overrides
    @property
    def goal_observation(self):
        """ Get the goal space part of the current observation. """
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env

        # FIXME: technically we need to invert the angle
        return self.transform_to_goal_space(obj.get_current_obs())

    def _append_goal_observation(self, obs):
        """ Append the current goal based observation to the given original observation. """
        if self.append_transformed_obs:
            return np.concatenate(
                [obs, np.array(self.transform_to_goal_space(obs)), np.array(self.current_goal)]
            )
        return np.concatenate([obs, np.array(self.current_goal)])

    @property
    @overrides
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @overrides
    def log_diagnostics(self, paths, n_traj=1, *args, **kwargs):
        # Process by time steps
        distances = [
            np.mean(path['env_infos']['distance'])
            for path in paths
            ]
        initial_goal_distances = [
            path['env_infos']['distance'][0] for path in paths
            ]
        reward_dist = [
            np.sum(path['env_infos']['reward_dist'])
            for path in paths
            ]
        reward_inner = [
            np.sum(path['env_infos']['reward_inner'])
            for path in paths
            ]
        goals = [path['observations'][0, -self.feasible_goal_space.flat_dim:] for path in paths]  # assumes const goal
        success = [int(np.min(path['env_infos']['distance']) <= self.dist_threshold) for path in paths]
        feasible = [int(self.feasible_goal_space.contains(goal)) for goal in goals]
        if n_traj > 1:
            avg_success = []
            for i in range(len(success) // n_traj):
                avg_success.append(np.mean(success[3 * i: 3 * i + 3]))
            success = avg_success  # here the success can be non-int

        print('the succes is: ', success)
        print('the feasible is: ', feasible)

        # Process by trajectories
        logger.record_tabular('InitGoalDistance', np.mean(initial_goal_distances))
        logger.record_tabular('MeanPathDistance', np.mean(distances))
        logger.record_tabular('AvgTotalRewardDist', np.mean(reward_dist))
        logger.record_tabular('AvgTotalRewardInner', np.mean(reward_inner))
        logger.record_tabular('SuccessRate', np.mean(success))
        logger.record_tabular('FeasibilityRate', np.mean(feasible))


def update_env_state_generator(env, goal_generator):
    """ Update the goal generator for normalized environment. """
    obj = env
    while not hasattr(obj, 'update_goal_generator') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    if hasattr(obj, 'update_goal_generator'):
        return obj.update_goal_generator(goal_generator)
    else:
        raise NotImplementedError('Unsupported environment')


def get_goal_observation(env):
    if hasattr(env, 'goal_observation'):
        return env.goal_observation  # should be unnecessary
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.goal_observation
    else:
        raise NotImplementedError('Unsupported environment')


def get_current_goal(env):
    if hasattr(env, 'current_goal'):
        return env.current_goal
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.current_goal
    else:
        raise NotImplementedError('Unsupported environment')


def generate_initial_goals(env, policy, goal_range, horizon=500, size=10000):
    current_goal = get_current_goal(env)
    goal_dim = np.array(current_goal).shape
    done = False
    obs = env.reset()
    goals = [get_goal_observation(env)]
    steps = 0
    while len(goals) < size:
        steps += 1
        if done or steps >= horizon:
            steps = 0
            done = False
            update_env_state_generator(
                env,
                FixedStateGenerator(
                    np.random.uniform(-goal_range, goal_range, goal_dim)
                )
            )
            obs = env.reset()
            goals.append(get_goal_observation(env))
        else:
            action, _ = policy.get_action(obs)
            obs, _, done, _ = env.step(action)
            goals.append(get_goal_observation(env))

    return np.array(goals)
