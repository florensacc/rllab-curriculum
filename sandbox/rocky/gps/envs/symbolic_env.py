from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env
from rllab.envs.normalized_env import NormalizedEnv


class SymbolicEnv(Env):
    def reward_sym(self, obs_var, action_var):
        raise NotImplementedError


class SymbolicNormalize(NormalizedEnv):
    def reward_sym(self, obs_var, action_var):
        lb, ub = self._wrapped_env.action_space.bounds
        scaled_action = lb + (action_var + 1.) * 0.5 * (ub - lb)
        return self._wrapped_env.reward_sym(obs_var, scaled_action)
