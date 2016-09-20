from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from cached_property import cached_property
import numpy as np


class PartialObsMazeEnv(Env):
    def __init__(self, env, vis_range=1):
        self.env = env
        self.vis_range = vis_range
        self.cached_desc = None

    @property
    def visited(self):
        if not hasattr(self, '_visited'):
            self._visited = np.zeros((self.n_row, self.n_col), dtype=np.bool)
        return self._visited

    def reset(self):
        self.env.reset()
        assert np.all(np.in1d(self.env.desc, ('W', 'S', 'F', 'G')))
        self.cached_desc = np.asarray(
            np.zeros((self.env.n_row + 2 * self.vis_range, self.env.n_col + 2 * self.vis_range)),
            dtype=self.env.desc.dtype,
        )
        self.cached_desc[:] = 'W'
        self.cached_desc[self.vis_range:-self.vis_range, self.vis_range:-self.vis_range] = self.env.desc
        self.update_visit()
        return self.get_current_obs()

    def update_visit(self):
        cur_x, cur_y = self.state / self.n_col, self.state % self.n_col
        self.visited[max(0, cur_x - 1):min(self.n_row, cur_x + 2), max(0, cur_y - 1):min(self.n_col, cur_y + 2)] = True

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        self.update_visit()
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

    def reset_trial(self):
        self.env.reset_trial()
        self.visited[:] = False
        return self.reset()

    @property
    def viewer(self):
        return self.env.viewer

    @viewer.setter
    def viewer(self, viewer):
        self.env.viewer = viewer

    @property
    def n_col(self):
        return self.env.n_col

    @property
    def n_row(self):
        return self.env.n_row

    @property
    def desc(self):
        return self.env.desc

    @property
    def state(self):
        return self.env.state

    def render(self, close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from sandbox.rocky.hrl.envs.gym_renderer import Viewer
        if self.viewer is None:
            self.viewer = Viewer(500, 500)
            self.viewer.set_bounds(-1, self.n_col + 1, -1, self.n_row + 1)
        for row_idx in range(self.n_row + 1):
            self.viewer.draw_line((0., row_idx), (self.n_col, row_idx))
        for col_idx in range(self.n_col + 1):
            self.viewer.draw_line((col_idx, 0.), (col_idx, self.n_row))
        for row_idx in range(self.n_row):
            for col_idx in range(self.n_col):
                entry = self.desc[row_idx][col_idx]
                if not self.visited[row_idx, col_idx]:
                    self.viewer.draw_polygon(
                        v=[
                            (col_idx, self.n_row - row_idx),
                            (col_idx, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx),
                        ],
                        color=(0.3, 0.3, 0.3)
                    )
                elif entry == 'F' or entry == 'S' or entry == 'G':
                    self.viewer.draw_polygon(
                        v=[
                            (col_idx, self.n_row - row_idx),
                            (col_idx, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx),
                        ],
                        color=(1, 1, 1)
                    )
                    if entry in ['S', 'G']:
                        self.viewer.draw_circle(
                            radius=0.25,
                            center=(col_idx + 0.5, self.n_row - row_idx - 0.5),
                            color=(0, 0, 1) if entry == 'S' else (1, 0, 0)
                        )
                elif entry == 'W':
                    self.viewer.draw_polygon(
                        v=[
                            (col_idx, self.n_row - row_idx),
                            (col_idx, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx - 1),
                            (col_idx + 1, self.n_row - row_idx),
                        ],
                        color=(0, 0, 0)
                    )
                block_size = 0.25
                if row_idx * self.n_col + col_idx == self.state:
                    self.viewer.draw_circle(
                        radius=0.25,
                        center=(col_idx + 0.5, self.n_row - row_idx - 0.5),
                        color=(0, 0, 0),
                    )
                self.viewer.draw_line((col_idx, self.n_row - row_idx), (col_idx, self.n_row - row_idx - 1))
                self.viewer.draw_line((col_idx, self.n_row - row_idx - 1), (col_idx + 1, self.n_row - row_idx - 1))
                self.viewer.draw_line((col_idx + 1, self.n_row - row_idx - 1), (col_idx + 1, self.n_row - row_idx))
                self.viewer.draw_line((col_idx + 1, self.n_row - row_idx), (col_idx, self.n_row - row_idx))

        self.viewer.render()
        self.viewer.window.dispatch_events()
        self.viewer.window.flip()
