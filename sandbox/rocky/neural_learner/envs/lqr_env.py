from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.base import Env
from rllab.spaces.box import Box

BIG = 50000


class LqrEnv(Env):

    def __init__(self, obs_dim, action_dim):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._observation_space = Box(low=-BIG, high=BIG, shape=(obs_dim,))
        self._action_dim = Box(low=)
        pass
