from __future__ import print_function
from __future__ import absolute_import
from rllab.spaces import Discrete
from rllab.envs.base import Env, Step
import numpy as np
import contextlib
import random
from .maze.dfs_maze_generator import DFSMazeGenerator


@contextlib.contextmanager
def set_seed_tmp(seed=None):
    if seed is None:
        yield
    else:
        state = random.getstate()
        np_state = np.random.get_state()
        random.seed(seed)
        np.random.seed(seed)
        yield
        np.random.set_state(np_state)
        random.setstate(state)


class RandomMazeEnv(Env):
    """
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    """

    def __init__(self, n_row, n_col, maze_gen=None, seed_pool=None):
        self.n_row = n_row
        self.n_col = n_col
        self.desc = None
        self.start_state = None
        self.state = None
        self.viewer = None
        if maze_gen is None:
            maze_gen = DFSMazeGenerator()
        self.maze_gen = maze_gen
        self.seed_pool = seed_pool
        self.reset_trial()

    def regenerate_map(self):
        grid = self.maze_gen.gen_maze(self.n_row, self.n_col)
        desc = np.asarray(np.zeros_like(grid), dtype='<U1')
        desc[grid == 0] = 'W'
        desc[grid == 1] = 'F'
        free_pos = list(zip(*np.where(desc == 'F')))
        start_position = random.choice(free_pos)
        goal_position = random.choice(list(set(free_pos) - {start_position}))
        desc[start_position] = 'S'
        desc[goal_position] = 'G'

        self.desc = desc
        # Now, set the starting and goal position
        (start_x,), (start_y,) = np.nonzero(desc == 'S')
        self.start_state = start_x * self.n_col + start_y
        self.state = None

    def reset_trial(self):
        if self.seed_pool is not None:
            seed = random.choice(self.seed_pool)
        else:
            seed = None
        with set_seed_tmp(seed):
            # Reinitialize the map
            self.regenerate_map()
            return self.reset()

    def reset(self):
        self.state = self.start_state
        return self.state

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

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
        possible_next_states = self.get_possible_next_states(self.state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]

        next_x = next_state // self.n_col
        next_y = next_state % self.n_col

        next_state_type = self.desc[next_x, next_y]
        # if next_state_type == 'H':
        #     done = True
        #     reward = -100#0
        success = 0
        if next_state_type in ['F', 'S']:
            done = False
            reward = -0.01  # 0
        elif next_state_type == 'G':
            done = True
            reward = 1#000#0  # 100
            success = 1
        else:
            raise NotImplementedError
        self.state = next_state
        return Step(observation=self.state, reward=reward, done=done, success=success)

    def get_possible_next_states(self, state, action):
        """
        Given the state and action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param state: start state
        :param action: action
        :return: a list of pairs (s', p(s'|s,a))
        """
        # assert self.observation_space.contains(state)
        # assert self.action_space.contains(action)

        x = state // self.n_col
        y = state % self.n_col
        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.n_row - 1, self.n_col - 1]
        )
        next_state = next_coords[0] * self.n_col + next_coords[1]
        state_type = self.desc[x, y]
        next_state_type = self.desc[next_coords[0], next_coords[1]]
        if next_state_type == 'W' or state_type == 'H' or state_type == 'G':
            return [(state, 1.)]
        else:
            return [(next_state, 1.)]

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Discrete(self.n_row * self.n_col)

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
                if entry == 'F' or entry == 'S' or entry == 'G':
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


if __name__ == "__main__":
    env = RandomMazeEnv(n_row=9, n_col=9)  # , seed=0)
    # env.reset_trial()
    while True:
        import time

        time.sleep(0.01)
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset_trial()
