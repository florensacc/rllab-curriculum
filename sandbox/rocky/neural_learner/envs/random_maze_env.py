from __future__ import print_function
from __future__ import absolute_import
from rllab.spaces import Discrete
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
import numpy as np
import contextlib
import random
from matplotlib import pyplot as plt
import matplotlib.cm as cm


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


def gen_maze(n_row, n_col):
    """
    Generate a maze of size (num_rows*2+1, num_cols*2+1)
    Entry:
    0: Wall
    1: Free
    """

    assert n_row % 2 == 1 and n_row >= 5
    assert n_col % 2 == 1 and n_col >= 5

    maze_n_row = n_row // 2
    maze_n_col = n_col // 2

    M = np.zeros((maze_n_row, maze_n_col, 5), dtype=np.uint8)
    # The array M is going to hold the array information for each cell.
    # The first four coordinates tell if walls exist on those sides
    # and the fifth indicates if the cell has been visited in the search.
    # M(LEFT, UP, RIGHT, DOWN, CHECK_IF_VISITED)
    # image = np.zeros((num_rows * 10, num_cols * 10), dtype=np.uint8)
    # The array image is going to be the output image to display

    # Set starting row and column
    r = 0
    c = 0
    history = [(r, c), (0, 0), (0, c), (r, 0)]  # The history is the

    # Trace a path though the cells of the maze and open walls along the path.
    # We do this with a while loop, repeating the loop until there is no history,
    # which would mean we backtracked to the initial start.
    while history:
        M[r, c, 4] = 1  # designate this location as visited
        # check if the adjacent cells are valid for moving to
        check = []
        if c > 0 and M[r, c - 1, 4] == 0:
            check.append('L')
        if r > 0 and M[r - 1, c, 4] == 0:
            check.append('U')
        if c < maze_n_col - 1 and M[r, c + 1, 4] == 0:
            check.append('R')
        if r < maze_n_row - 1 and M[r + 1, c, 4] == 0:
            check.append('D')

        if len(check):  # If there is a valid cell to move to.
            # Mark the walls between cells as open if we move
            history.append([r, c])
            move_direction = random.choice(check)
            if move_direction == 'L':
                M[r, c, 0] = 1
                c = c - 1
                M[r, c, 2] = 1
            if move_direction == 'U':
                M[r, c, 1] = 1
                r = r - 1
                M[r, c, 3] = 1
            if move_direction == 'R':
                M[r, c, 2] = 1
                c = c + 1
                M[r, c, 0] = 1
            if move_direction == 'D':
                M[r, c, 3] = 1
                r = r + 1
                M[r, c, 1] = 1
        else:  # If there are no valid cells to move to.
            # retrace one step back in history if no move is possible
            r, c = history.pop()

    # Open the walls at the start and finish
    M[0, 0, 0] = 1
    M[0, maze_n_col - 1, 1] = 1
    M[maze_n_row - 1, maze_n_col - 1, 2] = 1
    M[maze_n_row - 1, 0, 3] = 1

    grid = np.zeros((n_row, n_col))

    # Generate the image for display
    for row in range(0, maze_n_row):
        for col in range(0, maze_n_col):
            cell_data = M[row, col]
            grid[row * 2 + 1, col * 2 + 1] = 1
            if cell_data[0] == 1:
                grid[row * 2 + 1, col * 2] = 1
            if cell_data[1] == 1:
                grid[row * 2, col * 2 + 1] = 1
            if cell_data[2] == 1:
                grid[row * 2 + 1, col * 2 + 2] = 1
            if cell_data[3] == 1:
                grid[row * 2 + 2, col * 2 + 1] = 1

    return grid


class RandomMazeEnv(Env):
    """
    'S' : starting point
    'F' or '.': free space
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    """

    def __init__(self, n_row, n_col, seed_pool=None):
        self.n_row = n_row
        self.n_col = n_col
        self.desc = None
        self.start_state = None
        self.state = None
        self.viewer = None
        self.seed_pool = seed_pool
        self.reset_trial()

    def regenerate_map(self):
        grid = gen_maze(self.n_row, self.n_col)
        desc = np.asarray(np.zeros_like(grid), dtype='<U1')
        desc[grid == 0] = 'W'
        desc[grid == 1] = 'F'
        desc[1, 0] = 'S'
        goal_position = random.choice([(self.n_row - 1, 1), (self.n_row - 2, self.n_col - 1), (0, self.n_col - 2)])
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

        next_x = next_state / self.n_col
        next_y = next_state % self.n_col

        next_state_type = self.desc[next_x, next_y]
        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F', 'S']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError
        self.state = next_state
        return Step(observation=self.state, reward=reward, done=done)

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

    def render(self):
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
                        color=(255, 255, 255)
                    )
                    if entry in ['S', 'G']:
                        self.viewer.draw_circle(
                            radius=0.25,
                            center=(col_idx + 0.5, self.n_row - row_idx - 0.5),
                            color=(0, 0, 255) if entry == 'S' else (255, 0, 0)
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
                if row_idx * self.n_col + col_idx == self.state:
                    self.viewer.draw_circle(
                        radius=0.25,
                        center=(col_idx + 0.5, self.n_row - row_idx - 0.5),
                        color=(0, 0, 0),
                    )
        self.viewer.render()
        self.viewer.window.dispatch_events()
        self.viewer.window.flip()


if __name__ == "__main__":
    env = RandomMazeEnv(n_row=7, n_col=7, seed=0)
    # env.reset_trial()
    while True:
        import time

        time.sleep(0.01)
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        if done:
            env.reset()#reset_trial()
