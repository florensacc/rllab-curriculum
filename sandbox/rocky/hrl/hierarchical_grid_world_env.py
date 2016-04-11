from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.envs.grid_world_env import GridWorldEnv
import numpy as np


def expand_grid(high_grid, low_grid):
    """
    Construct a large grid where each cell in the high_grid is replaced by a copy of low_grid. The starting position
    will be respected in the following sense:
        - The starting position in the low_grid that's also inside the starting position in the high_grid will be the
        starting position of the total grid
        - All other starting positions in the low_grid will be replaced by a free grid ('F')
    :return: the expanded grid
    """
    high_grid = np.array(map(list, high_grid))
    low_grid = np.array(map(list, low_grid))
    high_n_row, high_n_col = high_grid.shape
    low_n_row, low_n_col = low_grid.shape

    total_n_row = high_n_row * low_n_row
    total_n_col = high_n_col * low_n_col

    start_free_low_grid = np.copy(low_grid)
    start_free_low_grid[start_free_low_grid == 'S'] = 'F'

    total_grid = np.zeros((total_n_row, total_n_col), high_grid.dtype)
    for row in xrange(high_n_row):
        for col in xrange(high_n_col):
            if high_grid[row, col] == 'S':
                total_grid[row * low_n_row:(row + 1) * low_n_row, col * low_n_col:(col + 1) * low_n_col] = low_grid
            else:
                total_grid[row * low_n_row:(row + 1) * low_n_row, col * low_n_col:(col + 1) * low_n_col] = \
                    start_free_low_grid
    return total_grid


class HierarchicalGridWorldEnv(Env, Serializable):
    def __init__(self, high_grid, low_grid):
        Serializable.quick_init(self, locals())
        self.high_grid = np.array(map(list, high_grid))
        self.low_grid = np.array(map(list, low_grid))

        self.high_n_row, self.high_n_col = self.high_grid.shape
        self.low_n_row, self.low_n_col = self.low_grid.shape

        self.total_grid = expand_grid(high_grid, low_grid)
        self.total_n_row, self.total_n_col = self.total_grid.shape
        self.flat_env = GridWorldEnv(self.total_grid)

        self._observation_space = Product(
            Discrete(self.high_n_row * self.high_n_col),
            Discrete(self.low_n_row * self.low_n_col),
        )
        self._action_space = Discrete(4)
        self.reset()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        flat_obs = self.flat_env.reset()
        return self._get_hierarchical_obs(flat_obs)

    def _get_hierarchical_obs(self, flat_obs):
        total_coord = flat_obs
        total_row = total_coord / self.total_n_col
        total_col = total_coord % self.total_n_col
        high_row = total_row / self.low_n_row
        low_row = total_row % self.low_n_row
        high_col = total_col / self.low_n_col
        low_col = total_col % self.low_n_col
        return (high_row * self.high_n_col + high_col, low_row * self.low_n_col + low_col)

    def step(self, action):
        next_obs, reward, done, info = self.flat_env.step(action)
        return Step(self._get_hierarchical_obs(next_obs), reward, done, **info)
