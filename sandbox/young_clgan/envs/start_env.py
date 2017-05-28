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

from sandbox.young_clgan.envs.base import StateGenerator, UniformListStateGenerator, \
    UniformStateGenerator, FixedStateGenerator, StateAuxiliaryEnv, update_env_state_generator


class StartEnv(StateAuxiliaryEnv):
    """ A wrapper of StateAuxiliaryEnv to make it compatible with the old goal env."""

    def __init__(self, start_generator=None, *args, **kwargs):
        if start_generator is not None:
            kwargs['state_generator'] = start_generator
        super(StartEnv, self).__init__(*args, **kwargs)

    def update_start_generator(self, *args, **kwargs):
        return self.update_state_generator(*args, **kwargs)
        
    def update_start(self, start=None, *args, **kwargs):
        return self.update_aux_state(state=start, *args, **kwargs)
        
    @property
    def start_generator(self):
        return self.state_generator
    
    @property
    def current_start(self):
        return self.current_aux_state


class StartExplorationEnv(StartEnv, ProxyEnv, Serializable):
    def __init__(self, env, start_generator, only_feasible=False, start_bounds=None, **kwargs):
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
        StartEnv.__init__(self, **kwargs)
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


# def generate_initial_starts(env, policy, start_range, start_center=None, horizon=500, size=10000):  # TODO: get starts
#     done = False
#     obs = env.reset()
#     starts = [env.get_current_obs()]
#     start_dim = np.array(starts[0]).shape
#     if start_center is None:
#         start_center = np.zeros(start_dim)
#     steps = 0
#     while len(starts) < size:
#         steps += 1
#         if done or steps >= horizon:
#             steps = 0
#             done = False
#             update_env_state_generator(
#                 env,
#                 FixedStateGenerator(
#                     start_center + np.random.uniform(-start_range, start_range, start_dim)
#                 )
#             )
#             obs = env.reset()
#             starts.append(env.get_current_obs())
#         else:
#             action, _ = policy.get_action(obs)
#             obs, _, done, _ = env.step(action)
#             starts.append(env.get_current_obs())
#
#     return np.array(starts)
#
#
# def update_env_start_generator(env, start_generator):
#     return update_env_state_generator(env, start_generator)
#
#
# def evaluate_start_env(env, policy, horizon, n_starts=10, n_traj=1, **kwargs):
#     paths = [rollout(env=env, agent=policy, max_path_length=horizon) for _ in range(int(n_starts))]
#     env.log_diagnostics(paths, n_traj=n_traj, **kwargs)

