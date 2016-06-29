from __future__ import print_function
from __future__ import absolute_import
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.spaces.box import Box
from rllab.envs.base import Step
import numpy as np

AGENT = 0
GOAL = 1
WALL = 2
HOLE = 3
N_OBJECT_TYPES = 4


class ImageGridWorld(GridWorldEnv):
    def __init__(self, desc):
        super(ImageGridWorld, self).__init__(desc)
        self._observation_space = Box(low=0., high=1., shape=(self.n_row, self.n_col, N_OBJECT_TYPES))
        self._original_obs_space = GridWorldEnv.observation_space.fget(self)

    @property
    def observation_space(self):
        return self._observation_space

    def reset(self):
        super(ImageGridWorld, self).reset()
        return self.get_current_obs()

    def step(self, action):
        _, reward, done, info = super(ImageGridWorld, self).step(action)
        agent_state = self._original_obs_space.flatten(self.state)
        return Step(self.get_current_obs(), reward, done, **dict(info, agent_state=agent_state))

    def get_current_obs(self):
        ret = np.zeros(self._observation_space.shape)
        ret[self.desc == 'H', HOLE] = 1
        ret[self.desc == 'W', WALL] = 1
        ret[self.desc == 'G', GOAL] = 1
        cur_x = self.state / self.n_col
        cur_y = self.state % self.n_col
        ret[cur_x, cur_y, AGENT] = 1
        return ret
