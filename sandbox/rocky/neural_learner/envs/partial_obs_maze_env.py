from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from cached_property import cached_property
import numpy as np


class PartialObsMazeEnv(Env):
    def __init__(self, env, vis_range=1):
        self.env = env
        self.vis_range = vis_range
        self.cached_desc = None

    def reset(self):
        self.env.reset()
        assert np.all(np.in1d(self.env.desc, ('W', 'S', 'F', 'G')))
        self.cached_desc = np.asarray(
            np.zeros((self.env.n_row + 2 * self.vis_range, self.env.n_col + 2 * self.vis_range)),
            dtype=self.env.desc.dtype,
        )
        self.cached_desc[:] = 'W'
        self.cached_desc[self.vis_range:-self.vis_range, self.vis_range:-self.vis_range] = self.env.desc
        return self.get_current_obs()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return Step(self.get_current_obs(), reward, done, **info)

    @cached_property
    def observation_space(self):
        return Box(low=0, high=1, shape=(self.vis_range * 2 + 1, self.vis_range * 2 + 1, 3))

    @cached_property
    def action_space(self):
        return self.env.action_space

    def get_current_obs(self):
        cur_x = self.env.state // self.env.n_col
        cur_y = self.env.state % self.env.n_col
        sliced = self.cached_desc[
                 cur_x + 1 - self.vis_range:cur_x + 1 + self.vis_range + 1,
                 cur_y + 1 - self.vis_range:cur_y + 1 + self.vis_range + 1,
                 ]
        obs = np.zeros((self.vis_range * 2 + 1, self.vis_range * 2 + 1, 3))
        obs[:, :, 0][sliced == 'F'] = 1
        obs[:, :, 0][sliced == 'S'] = 1
        obs[:, :, 1][sliced == 'W'] = 1
        obs[:, :, 2][sliced == 'G'] = 1
        return obs
        # import ipdb;
        # ipdb.set_trace()
        # pass

    def reset_trial(self):
        self.env.reset_trial()
        return self.reset()
