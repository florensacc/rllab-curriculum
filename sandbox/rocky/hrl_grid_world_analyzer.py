import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from rllab.misc.tabulate import tabulate
from rllab.sampler.utils import rollout
from rllab.spaces.product import Product
import joblib


class GridPlot(object):
    def __init__(self, grid_size, title=None):
        plt.figure(figsize=(5, 5))
        ax = plt.axes(aspect=1)

        plt.tick_params(axis='x', labelbottom='off')
        plt.tick_params(axis='y', labelleft='off')
        if title:
            plt.title(title)

        ax.set_xticks(range(grid_size + 1))
        ax.set_yticks(range(grid_size + 1))
        ax.grid(True, linestyle='-', color=(0, 0, 0), alpha=1, linewidth=1)
        self._grid_size = grid_size
        self._ax = ax

    def add_text(self, x, y, text, gravity='center', size=10):
        # transform the grid index to coordinate
        x, y = y, self._grid_size - 1 - x
        if gravity == 'center':
            self._ax.text(x + 0.5, y + 0.5, text, ha='center', va='center', size=size)
        elif gravity == 'left':
            self._ax.text(x + 0.05, y + 0.5, text, ha='left', va='center', size=size)
        elif gravity == 'top':
            self._ax.text(x + 0.5, y + 1 - 0.05, text, ha='center', va='top', size=size)
        elif gravity == 'right':
            self._ax.text(x + 1 - 0.05, y + 0.5, text, ha='right', va='center', size=size)
        elif gravity == 'bottom':
            self._ax.text(x + 0.5, y + 0.05, text, ha='center', va='bottom', size=size)
        else:
            raise NotImplementedError

    def color_grid(self, x, y, color, alpha=1.):
        x, y = y, self._grid_size - 1 - x
        self._ax.add_patch(patches.Rectangle(
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

    def __init__(self, file_name):
        print "loading"
        import sys
        sys.stdout.flush()
        params = joblib.load(file_name)
        print "loaded"
        sys.stdout.flush()
        self._env = env = params["env"]
        self._policy = policy = params["policy"]
        self._n_row = env.wrapped_env.n_row
        self._n_col = env.wrapped_env.n_col
        if isinstance(env.observation_space, Product):
            pass
        else:
            self._n_states = env.observation_space.n
            self._n_actions = env.action_space.n
            self._n_subgoals = policy.subgoal_space.n
            self._obs_space = env.observation_space
            self._action_space = env.action_space
            self._subgoal_space = policy.subgoal_space
            self._high_obs_space = policy.high_env_spec.observation_space
            self._low_obs_space = policy.low_env_spec.observation_space
            self._subgoal_interval = policy.subgoal_interval

    def analyze(self):
        self.print_subgoals()
        self.print_state_visitation_frequency()
        self.print_marginalized_policy()

    def print_subgoals(self):
        """
        Print the probability of the low-level policy taking each action for each state, subgoal, and time step
        """
        for x in range(self._n_row):
            for y in range(self._n_col):
                state = x * self._n_col + y
                # for state in range(env.observation_space.n):
                print "State (%d,%d)" % (x, y)
                tabulate_data = []
                high_obs = self._high_obs_space.flatten(state)
                high_prob = self._policy.high_policy.dist_info([high_obs], None)['prob'].flatten()
                row = [""]
                for subgoal in range(self._n_subgoals):
                    row.append("Subgoal %d: p=%.2f" % (subgoal, high_prob[subgoal]))
                tabulate_data.append(row)
                for time_step in range(self._subgoal_interval):
                    row = ["T=%d" % (time_step)]
                    for subgoal in range(self._n_subgoals):
                        low_obs = self._low_obs_space.flatten((state, subgoal, time_step))
                        prob = self._policy.low_policy.dist_info([low_obs], None)['prob'].flatten()
                        row.append(" ".join(["%.2f" % px for px in prob]))
                    tabulate_data.append(row)
                print tabulate(tabulate_data)
                print ""

    def print_marginalized_policy(self):
        marginal_policy = np.zeros((self._n_states, self._n_actions))
        for state in range(self._n_states):
            goal_probs = self._policy.high_policy.get_action(state)[1]["prob"]
            for action in range(self._n_actions):
                action_sequence = self._env._action_map[action]
                # print action_sequence
                total_prob = 0.
                for goal in range(self._n_subgoals):
                    goal_prob = goal_probs[goal]
                    for time_step in range(self._policy.subgoal_interval):
                        goal_prob *= self._policy.low_policy.get_action((state, goal, time_step))[1]["prob"][
                            action_sequence[time_step]]
                    total_prob += goal_prob
                marginal_policy[state, action] = total_prob

        grid_plot = GridPlot(4, title="Marginal Policy Plot")
        for x in range(self._n_row):
            for y in range(self._n_col):
                state = x * self._n_col + y
                for action in range(self._n_actions):
                    grid_plot.add_text(x, y, text="%.2f" % marginal_policy[state, action], gravity=self.ACTION_MAP[action])
        # color the starting position
        grid_plot.color_grid(0, 0, color='b', alpha=0.1)

    def print_state_visitation_frequency(self):
        paths = []
        for _ in xrange(50):
            paths.append(rollout(env=self._env, agent=self._policy, max_length=200))
        states = np.vstack([p["observations"] for p in paths])
        print np.array_str(np.mean(states, axis=0)[:16].reshape((4, 4)))

    def rollout(self, max_length=100):
        path = rollout(env=self._env, agent=self._policy, max_length=max_length)
        obs = [self._obs_space.unflatten(x) for x in path["observations"]]
        actions = [self._action_space.unflatten(x) for x in path["actions"]]
        subgoals = [self._subgoal_space.unflatten(x) for x in path["agent_infos"]["subgoal"]]
        for o, a, g in zip(obs, actions, subgoals):
            print "At state %d, subgoal %d, took action %s" % (o, g, self.ACTION_MAP[a])

