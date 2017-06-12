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

from sandbox.young_clgan.envs.goal_env import GoalExplorationEnv
from sandbox.young_clgan.envs.start_env import StartEnv, StartExplorationEnv


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

