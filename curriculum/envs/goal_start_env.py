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

from curriculum.envs.goal_env import GoalExplorationEnv
from curriculum.envs.start_env import StartEnv, StartExplorationEnv


class GoalStartExplorationEnv(StartEnv, GoalExplorationEnv, Serializable):
    def __init__(self, *args, **kwargs):
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
        StartEnv.__init__(self, *args, **kwargs)
        GoalExplorationEnv.__init__(self, *args, **kwargs)

    @overrides
    def reset(self, init_state=None, *args, **kwargs): # this does NOT call the full reset of StartEnvExploration!
        self.update_start(*args, **kwargs)
        if init_state is None:
            GoalExplorationEnv.reset(self, init_state=self.current_start, *args, **kwargs)
        else:
            GoalExplorationEnv.reset(self, init_state=init_state, *args, **kwargs)
        return self.get_current_obs()

    @overrides
    def step(self, action):
        obs, r, d, info = GoalExplorationEnv.step(self, action)
        return (
            StartEnv.append_start_observation(self, obs), r, d, info
        )

    @overrides
    def get_current_obs(self):
        goal_obs = GoalExplorationEnv.get_current_obs(self)
        return StartEnv.append_start_observation(self, goal_obs)

    @overrides
    def transform_to_start_space(self, obs, *args, **kwargs):
        obj = self.wrapped_env
        while not hasattr(obj, 'transform_to_start_space') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        if hasattr(obj, 'transform_to_start_space'):
            return obj.transform_to_start_space(obs, *args, **kwargs) # but the goal might not leave in the same space!
        else:
            return super(GoalStartExplorationEnv, self).transform_to_start_space(obs, *args, **kwargs)

    @overrides
    @property  # this is needed for the reset! When we reset we don't have the env_infos..
    def start_observation(self):
        obj = self.wrapped_env
        while not hasattr(obj, 'start_observation') and hasattr(obj, 'wrapped_env'):
            obj = obj.wrapped_env
        if hasattr(obj, 'start_observation'):
            return obj.start_observation # but the goal might not leave in the same space!
        else:
            return self.transform_to_start_space(obj.get_current_obs())


