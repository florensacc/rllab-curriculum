from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.base import Env
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
import numpy as np


class HierarchicalGridWorldEnv(Env, Serializable):
    def __init__(self, high_grid, low_grid):
        Serializable.quick_init(self, locals())
        self.high_grid = np.array(map(list, high_grid))
        self.low_grid = np.array(map(list, low_grid))
        # self._desc = np.array(map(list, desc))
        self.high_n_row, self.high_n_col = high_grid.shape
        self.low_n_row, self.low_n_col = low_grid.shape

        (self.high_start_x,), (self.high_start_y,) = np.nonzero(self.high_grid == 'S')
        (self.low_start_x,), (self.low_start_y,) = np.nonzero(self.low_grid == 'S')

        self._start_state = ((self.high_start_x, self.high_start_y), (self.low_start_x, self.low_start_y))
        self._state = None

        self._observation_space = Product(
            Discrete(self.high_n_row * self.high_n_col),
            Discrete(self.low_n_row * self.low_n_col),
        )
        self._action_space = Discrete(4)
        self.reset()
        # self._n_row = n_row
        # self._n_col = n_col
        # (start_x,), (start_y,) = np.nonzero(self._desc == 'S')
        # self._start_state = np.array([start_x, start_y])
        # self._start_state.flags.writeable = False

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        self._state = self._start_state
        return self._get_current_obs()

    def _get_current_obs(self):
        (high_x, high_y), (low_x, low_y) = self._state
        high_pos = self.high_n_col * high_x + high_y
        low_pos = self.low_n_col * low_x + low_y
        return (high_pos, low_pos)

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        # action_idx = special.from_onehot(action)
        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
