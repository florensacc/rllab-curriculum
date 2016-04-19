from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from rllab.misc.tabulate import tabulate
from rllab.misc import special
from rllab.sampler.utils import rollout
from rllab.spaces.product import Product
import joblib


class GridPlot(object):
    def __init__(self, n_rows, n_cols, title=None):
        min_row_col = min(n_rows, n_cols)
        max_row_col = max(n_rows, n_cols)
        plt.figure(figsize=(5.0 / min_row_col * n_cols, 5.0 / min_row_col * n_rows))
        ax = plt.axes(aspect=1)

        plt.tick_params(axis='x', labelbottom='off')
        plt.tick_params(axis='y', labelleft='off')
        if title:
            plt.title(title)

        ax.set_xticks(range(n_cols + 1))
        ax.set_yticks(range(n_rows + 1))
        ax.grid(True, linestyle='-', color=(0, 0, 0), alpha=1, linewidth=1)
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.ax = ax

    def add_text(self, x, y, text, gravity='center', size=10):
        # transform the grid index to coordinate
        x, y = y, self.n_rows - 1 - x
        if gravity == 'center':
            self.ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', size=size)
        elif gravity == 'left':
            self.ax.text(x + 0.05, y + 0.5, text, ha='left', va='center', size=size)
        elif gravity == 'top':
            self.ax.text(x + 0.5, y + 1 - 0.05, text, ha='center', va='top', size=size)
        elif gravity == 'right':
            self.ax.text(x + 1 - 0.05, y + 0.5, text, ha='right', va='center', size=size)
        elif gravity == 'bottom':
            self.ax.text(x + 0.5, y + 0.05, text, ha='center', va='bottom', size=size)
        else:
            raise NotImplementedError

    def color_grid(self, x, y, color, alpha=1.):
        x, y = y, self.n_rows - 1 - x
        self.ax.add_patch(patches.Rectangle(
            (x, y),
            1,
            1,
            facecolor=color,
            alpha=alpha
        ))


class HrlAnalyzer(object):
    ACTION_MAP = [
        'left',
        'bottom',
        'right',
        'top'
    ]

    def __init__(self, file_name=None, params=None):
        if params is None:
            print("loading")
            import sys
            sys.stdout.flush()
            params = joblib.load(file_name)
            print("loaded")
            sys.stdout.flush()
        self.env = env = params["env"]
        self.policy = policy = params["policy"]
        self.observation_space = env.observation_space
        self.component_space = env.observation_space.components[0]
        # self._n_row = env.wrapped_env.n_row
        # self._n_col = env.wrapped_env.n_col
        # if isinstance(env.observation_space, Product):
        #     pass
        # else:
        #     self._n_states = env.observation_space.n
        #     self._n_actions = env.action_space.n
        #     self._n_subgoals = policy.subgoal_space.n
        #     self._obs_space = env.observation_space
        #     self._action_space = env.action_space
        #     self._subgoal_space = policy.subgoal_space
        #     self._high_obs_space = policy.high_env_spec.observation_space
        #     self._low_obs_space = policy.low_env_spec.observation_space
        #     self._subgoal_interval = policy.subgoal_interval

    def analyze(self):
        self.print_state_visitation_frequency()


    def print_state_visitation_frequency(self):
        paths = []
        for _ in xrange(50):
            paths.append(rollout(env=self.env, agent=self.policy, max_path_length=100))
        observations = np.vstack([p["observations"] for p in paths])
        self.print_total_frequency(observations)
        self.print_high_frequency(observations)

    def to_total_onehot(self, obs):
        high_coord_idx, low_coord_idx = obs
        high_row = high_coord_idx / self.env.high_n_col
        high_col = high_coord_idx % self.env.high_n_col
        low_row = low_coord_idx / self.env.low_n_col
        low_col = low_coord_idx % self.env.low_n_col
        total_row = high_row * self.env.low_n_row + low_row
        total_col = high_col * self.env.low_n_col + low_col
        return special.to_onehot(total_row * self.env.total_n_col + total_col, self.env.total_n_col *
                                 self.env.total_n_row)

    def print_total_frequency(self, observations):
        total_obs = map(self.observation_space.unflatten, observations)
        total_onehots = map(self.to_total_onehot, total_obs)
        mean_onehots = np.mean(total_onehots, axis=0).reshape(
            (self.env.total_n_row, self.env.total_n_col)
        )
        print(np.array2string(mean_onehots, formatter={'float_kind':lambda x: "%.2f" % x}))

    def print_high_frequency(self, observations):
        high_states = np.array(
            [self.component_space.flatten(self.observation_space.unflatten(x)[0]) for x in observations])
        mean_high_states = np.mean(high_states, axis=0).reshape(
            (self.env.high_n_row, self.env.high_n_col)
        )
        print(np.array2string(mean_high_states, formatter={'float_kind':lambda x: "%.2f" % x}))
