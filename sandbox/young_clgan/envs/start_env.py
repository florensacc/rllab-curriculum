"""
Start based environments. The classes inside this file should inherit the classes
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
import time

from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.sampler.utils import rollout
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides

from sandbox.young_clgan.envs.base import StateGenerator, UniformListStateGenerator, \
    UniformStateGenerator, FixedStateGenerator, StateAuxiliaryEnv, update_env_state_generator


class StartEnv(Serializable):
    """ A wrapper of StateAuxiliaryEnv to make it compatible with the old goal env."""

    def __init__(self, start_generator=None, append_start=False, obs2start_transform=None, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self._start_holder = StateAuxiliaryEnv(state_generator=start_generator, *args, **kwargs)
        self.append_start = append_start
        if obs2start_transform is None:
            self._obs2start_transform = lambda x: x
        else:
            self._obs2start_transform = obs2start_transform

    def transform_to_start_space(self, obs):
        """ Apply the start space transformation to the given observation. """
        return self._obs2start_transform(obs)

    def update_start_generator(self, *args, **kwargs):

        # print("updating start generator with ", *args, **kwargs)

        return self._start_holder.update_state_generator(*args, **kwargs)
        
    def update_start(self, start=None, *args, **kwargs):
        return self._start_holder.update_aux_state(state=start, *args, **kwargs)
        
    @property
    def start_generator(self):
        return self._start_holder.state_generator
    
    @property
    def current_start(self):
        return self._start_holder.current_aux_state

    @property
    def start_observation(self):
        """ Get the start space part of the current observation. """
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env
        return self.transform_to_start_space(obj.get_current_obs())

    def append_start_observation(self, obs):
        """ Append current start to the given original observation """
        if self.append_start:
            return np.concatenate([obs, np.array(self.current_start)])
        else:
            return obs

    def __getstate__(self):
        d = super(StartEnv, self).__getstate__()
        d['__start_holder'] = self._start_holder
        return d

    def __setstate__(self, d):
        super(StartEnv, self).__setstate__(d)
        self._start_holder = d['__start_holder']


class StartExplorationEnv(StartEnv, ProxyEnv, Serializable):
    def __init__(self, env, start_generator, only_feasible=False, start_bounds=None, *args, **kwargs):
        """
        This environment wraps around a normal environment to facilitate goal based exploration.
        Initial position based experiments should not use this class.
        
        :param env: wrapped env
        :param start_generator: a StateGenerator object
        :param obs_transform: a callable that transforms an observation of the wrapped environment into goal space
        :param terminal_eps: a threshold of distance that determines if a goal is reached
        :param terminate_env: a boolean that controls if the environment is terminated with the goal is reached
        :param start_bounds: array marking the UB of the rectangular limit of goals.
        :param distance_metric: L1 or L2 or a callable func
        :param goal_weight: coef of the goal based reward
        :param inner_weight: coef of the inner environment reward
        :param append_transformed_obs: append the transformation of the current observation to full observation
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        StartEnv.__init__(self, *args, **kwargs)
        self.update_start_generator(start_generator)
        
        self.start_bounds = start_bounds
        self.only_feasible = only_feasible
        
        # # TODO fix this
        # if self.start_bounds is None:
        #     self.start_bounds = self.wrapped_env.observation_space.bounds[1]  # we keep only UB
        #     self._feasible_start_space = self.wrapped_env.observation_space
        # else:
        #     self._feasible_start_space = Box(low=-1 * self.start_bounds, high=self.start_bounds)

    # @property
    # @overrides
    # def feasible_start_space(self):
    #     return self._feasible_start_space
    #
    # def is_feasible(self, start):
    #     obj = self.wrapped_env
    #     while not hasattr(obj, 'is_feasible') and hasattr(obj, 'wrapped_env'):
    #         obj = obj.wrapped_env
    #     if hasattr(obj, 'is_feasible'):
    #         return obj.is_feasible(np.array(start))  # but the goal might not leave in the same space!
    #     else:
    #         return True

    def reset(self, *args, **kwargs):
        self.update_start(*args, **kwargs)
        return self.wrapped_env.reset(init_state=self.current_start)


def generate_starts(env, policy=None, starts=None, horizon=50, size=10000, subsample=None, variance=1,
                    animated=False, speedup=1):
    """ If policy is None, brownian motion applied """
    if starts is None or len(starts) == 0:
        starts = [env.reset()]
    n_starts = len(starts)
    i = 0
    done = False
    obs = env.reset(init_state=starts[i % n_starts])
    states = [env.start_observation]
    steps = 0
    noise = 0
    if animated:
        env.render()
    while len(states) < size:
        steps += 1
        if done or steps >= horizon:
            steps = 0
            i += 1
            done = False
            obs = env.reset(init_state=starts[i % n_starts])
            states.append(env.start_observation)
        else:
            noise += np.random.randn(env.action_space.flat_dim) * variance
            if policy:
                action, _ = policy.get_action(obs)
            else:
                action = noise
            obs, _, _, _ = env.step(action)  # we don't care about done, otherwise will never advance!
            states.append(env.start_observation)
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    if subsample is None:
        return np.array(states)
    else:
        return np.array(states)[np.random.choice(np.shape(states)[0], size=subsample)]


def update_env_start_generator(env, start_generator):
    return update_env_state_generator(env, start_generator)
#
#
# def evaluate_start_env(env, policy, horizon, n_starts=10, n_traj=1, **kwargs):
#     paths = [rollout(env=env, agent=policy, max_path_length=horizon) for _ in range(int(n_starts))]
#     env.log_diagnostics(paths, n_traj=n_traj, **kwargs)

