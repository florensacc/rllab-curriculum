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
    Construct a large grid where each cell in the high_grid is replaced by a copy of low_grid. The starting and
    goal positions will be respected in the following sense:
        - The starting / goal positions in the low_grid that's also inside the starting position in the high_grid will
        be the starting / goal position of the total grid
        - All other starting / goal positions in the low_grid will be replaced by a free grid ('F')
    For other types of grids:
        - Wall and hole grids in the high grid will be replaced by an entire block of wall / hole grids
    :return: the expanded grid
    """
    high_grid = np.array(map(list, high_grid))
    low_grid = np.array(map(list, low_grid))
    high_n_row, high_n_col = high_grid.shape
    low_n_row, low_n_col = low_grid.shape

    total_n_row = high_n_row * low_n_row
    total_n_col = high_n_col * low_n_col

    start_only_low_grid = np.copy(low_grid)
    start_only_low_grid[start_only_low_grid == 'G'] = 'F'

    goal_only_low_grid = np.copy(low_grid)
    goal_only_low_grid[goal_only_low_grid == 'S'] = 'F'

    free_only_low_grid = np.copy(low_grid)
    free_only_low_grid[np.any([free_only_low_grid == 'S', free_only_low_grid == 'G'], axis=0)] = 'F'

    total_grid = np.zeros((total_n_row, total_n_col), high_grid.dtype)
    for row in xrange(high_n_row):
        for col in xrange(high_n_col):
            cell = high_grid[row, col]
            if cell == 'S':
                replace_grid = start_only_low_grid
            elif cell == 'G':
                replace_grid = goal_only_low_grid
            elif cell == 'F':
                replace_grid = free_only_low_grid
            elif cell == 'W':
                replace_grid = 'W'
            elif cell == 'H':
                replace_grid = 'H'
            else:
                raise NotImplementedError
            total_grid[row * low_n_row:(row + 1) * low_n_row, col * low_n_col:(col + 1) * low_n_col] = replace_grid
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
