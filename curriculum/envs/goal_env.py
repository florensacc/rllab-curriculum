"""
Goal based environments. The classes inside this file should inherit the classes
from the state environment base classes.
"""


import random
from rllab import spaces
import sys
import os.path as osp

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
from rllab.sampler.utils import rollout
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides

from curriculum.envs.base import StateGenerator, UniformListStateGenerator, \
    UniformStateGenerator, FixedStateGenerator, StateAuxiliaryEnv


class GoalEnv(Serializable):
    """ A wrapper of StateAuxiliaryEnv to make it compatible with the old goal env."""

    def __init__(self, goal_generator=None, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self._goal_holder = StateAuxiliaryEnv(state_generator=goal_generator, *args, **kwargs)

    def update_goal_generator(self, *args, **kwargs):
        return self._goal_holder.update_state_generator(*args, **kwargs)
        
    def update_goal(self, goal=None, *args, **kwargs):
        return self._goal_holder.update_aux_state(state=goal, *args, **kwargs)
        
    @property
    def goal_generator(self):
        return self._goal_holder.state_generator
    
    @property
    def current_goal(self):
        return self._goal_holder.current_aux_state

    def __getstate__(self):
        d = super(GoalEnv, self).__getstate__()
        d['__goal_holder'] = self._goal_holder
        return d

    def __setstate__(self, d):
        super(GoalEnv, self).__setstate__(d)
        self._goal_holder = d['__goal_holder']


class GoalExplorationEnv(GoalEnv, ProxyEnv, Serializable):
    def __init__(self, env, goal_generator, obs2goal_transform=None, terminal_eps=0.05, only_feasible=False,
                 terminate_env=False, goal_bounds=None, distance_metric='L2', extend_dist_rew=0., goal_weight=1,
                 inner_weight=0, append_transformed_obs=False, append_goal_to_observation=True, **kwargs):
        """
        This environment wraps around a normal environment to facilitate goal based exploration.
        Initial position based experiments should not use this class.
        
        :param env: wrapped env
        :param goal_generator: a StateGenerator object
        :param obs2goal_transform: a callable that transforms an observation of the wrapped environment into goal space
        :param terminal_eps: a threshold of distance that determines if a goal is reached
        :param terminate_env: a boolean that controls if the environment is terminated with the goal is reached
        :param goal_bounds: array marking the UB of the rectangular limit of goals.
        :param distance_metric: L1 or L2 or a callable func
        :param goal_weight: coef of the goal based reward
        :param inner_weight: coef of the inner environment reward
        :param append_transformed_obs: append the transformation of the current observation to full observation
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        GoalEnv.__init__(self, **kwargs)
        self.update_goal_generator(goal_generator)
        
        if obs2goal_transform is None:
            self._obs2goal_transform = lambda x: x   # needed for replay old policies [:2]
        else:
            self._obs2goal_transform = obs2goal_transform
        
        self.terminate_env = terminate_env
        self.goal_bounds = goal_bounds
        self.terminal_eps = terminal_eps
        self.only_feasible = only_feasible
        
        self.distance_metric = distance_metric
        self.extend_dist_rew_weight = extend_dist_rew
        self.goal_weight = goal_weight
        self.inner_weight = inner_weight
        
        self.append_transformed_obs = append_transformed_obs
        self.append_goal_to_observation = append_goal_to_observation

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

    def is_feasible(self, goal):
        obj = self.wrapped_env
        while not hasattr(obj, 'is_feasible') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        if hasattr(obj, 'is_feasible'):
            return obj.is_feasible(np.array(goal))  # but the goal might not leave in the same space!
        else:
            return True

    def reset(self, reset_goal=True, **kwargs):  # allows to pass init_state if needed
        if reset_goal:
            self.update_goal()
        #default behavior
        if self.append_goal_to_observation:
            ret = self.append_goal_observation(ProxyEnv.reset(self, goal=self.current_goal, **kwargs))  # the wrapped env needs to use or ignore it
        else:
            ret = ProxyEnv.reset(self, goal=self.current_goal, **kwargs)
        # used by disk environment # todo: make more generalizable!  NOT USABLE FOR OTHER ENVS!!
        if 'init_state' in kwargs and len(kwargs['init_state']) == 9:
            delta = tuple(kwargs['init_state'][-2:])  # joint position is in terms of amount moved
            original_goal = self.wrapped_env.model.data.site_xpos[-1]
            new_goal = delta[0] + original_goal[0], delta[1] + original_goal[1], original_goal[2] # z dim unchanged
            self.update_goal(new_goal)
        return ret

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        # print(reward_inner)
        if 'distance' not in info:
            info['distance'] = dist = self.dist_to_goal(observation)
            info['reward_dist'] = reward_dist = self.compute_dist_reward(observation)
            info['goal_reached'] = 1.0 * self.is_goal_reached(observation)
        else:
            # modified so that inner environment can pass in goal via step
            dist = info['distance']
            info['goal_reached'] = 1.0 * (dist < self.terminal_eps)
            info['reward_dist'] = reward_dist = - self.extend_dist_rew_weight * dist

        info['goal'] = self.current_goal
        # print(reward_dist)
        # print(reward_inner)
        # print("step: obs={}, goal={}, dist={}".format(self.append_goal_observation(observation), self.current_goal, dist))
        if self.terminate_env and info['goal_reached']:
            done = True
        if self.append_goal_to_observation:
            # print("appending goal to obs")
            observation = self.append_goal_observation(observation)
        return (
            observation,
            reward_dist + reward_inner + info['goal_reached'] * self.goal_weight,
            done,
            info
        )
        
    def is_goal_reached(self, observation):
        """ Return a boolean whether the (unaugmented) observation reached the goal. """
        if self.only_feasible:
            return self.dist_to_goal(observation) < self.terminal_eps and self.is_feasible(self.current_goal)
        else:
            return self.dist_to_goal(observation) < self.terminal_eps

    def compute_dist_reward(self, observation):
        """ Compute the 0 or 1 reward for reaching the goal. """
        return - self.extend_dist_rew_weight * self.dist_to_goal(observation)

    def dist_to_goal(self, obs):
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
        return self._obs2goal_transform(obs)
    
    def get_current_obs(self):
        """ Get the full current observation. The observation should be identical to the one used by policy. """
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env
        if self.append_goal_to_observation:
            return self.append_goal_observation(obj.get_current_obs())
        else:
            return obj.get_current_obs()

    @overrides
    @property
    def goal_observation(self):
        """ Get the goal space part of the current observation. """
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env

        # FIXME: technically we need to invert the angle
        return self.transform_to_goal_space(obj.get_current_obs())

    def append_goal_observation(self, obs):
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
        final_goal_distances = [
            path['env_infos']['distance'][-1] for path in paths
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
        success = [np.max(path['env_infos']['goal_reached']) for path in paths]
        feasible = [int(self.feasible_goal_space.contains(goal)) for goal in goals]
        if n_traj > 1:
            avg_success = []
            for i in range(len(success) // n_traj):
                avg_success.append(np.mean(success[3 * i: 3 * i + 3]))
            success = avg_success  # here the success can be non-int

        print('the mean success is: ', np.mean(success))
        print('the mean feasible is: ', np.mean(feasible))

        # Process by trajectories
        logger.record_tabular('AvgInitGoalDistance', np.mean(initial_goal_distances))
        logger.record_tabular('AvgFinalGoalDistance', np.mean(final_goal_distances))
        logger.record_tabular('MinFinalGoalDistance', np.min(final_goal_distances))
        logger.record_tabular('MeanPathDistance', np.mean(distances))
        logger.record_tabular('AvgTotalRewardDist', np.mean(reward_dist))
        logger.record_tabular('AvgTotalRewardInner', np.mean(reward_inner))
        logger.record_tabular('SuccessRate', np.mean(success))
        logger.record_tabular('FeasibilityRate', np.mean(feasible))


def get_goal_observation(env):
    if hasattr(env, 'goal_observation'):
        return env.goal_observation  # should be unnecessary
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.goal_observation
    else:
        raise NotImplementedError('Unsupported environment')


def get_current_goal(env):
    """ Get the current goal for the wrapped environment. """
    if hasattr(env, 'current_goal'):
        return env.current_goal
    elif hasattr(env, 'wrapped_env'):
        return env.wrapped_env.current_goal
    else:
        raise NotImplementedError('Unsupported environment')


def generate_initial_goals(env, policy, goal_range, goal_center=None, horizon=500, size=10000):
    current_goal = get_current_goal(env)
    goal_dim = np.array(current_goal).shape
    done = False
    obs = env.reset()
    goals = [get_goal_observation(env)]
    if goal_center is None:
        goal_center = np.zeros(goal_dim)
    steps = 0
    while len(goals) < size:
        steps += 1
        if done or steps >= horizon:
            steps = 0
            done = False
            env.update_goal_generator(
                FixedStateGenerator(
                    goal_center + np.random.uniform(-goal_range, goal_range, goal_dim)
                )
            )
            obs = env.reset()
            goals.append(get_goal_observation(env))
        else:
            action, _ = policy.get_action(obs)
            obs, _, done, _ = env.step(action)
            goals.append(get_goal_observation(env))

    return np.array(goals)


def generate_brownian_goals(env, starts=None, horizon=100, size=1000):
    current_goal = get_current_goal(env)
    if starts is None:
        starts = [current_goal]
    n_starts = len(starts)
    i = 0
    done = False
    env.reset(init_state=starts[i])
    goals = [get_goal_observation(env)]
    steps = 0
    while len(goals) < size:  # we ignore if it's done or not: the brownian motion around the goal will be short!
        steps += 1
        if done or steps >= horizon:
            steps = 0
            i += 1
            done = False
            env.reset(init_state=starts[i % n_starts])
            goals.append(get_goal_observation(env))
        else:
            action = np.random.randn(env.action_space.flat_dim)
            obs, _, done, _ = env.step(action)
            goals.append(get_goal_observation(env))

    return np.array(goals)

def evaluate_goal_env(env, policy, horizon, n_goals=10, n_traj=1, **kwargs):
    paths = [rollout(env=env, agent=policy, max_path_length=horizon) for _ in range(int(n_goals))]
    env.log_diagnostics(paths, n_traj=n_traj, **kwargs)

