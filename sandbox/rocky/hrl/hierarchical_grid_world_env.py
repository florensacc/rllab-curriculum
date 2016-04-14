from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.envs.grid_world_env import GridWorldEnv
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy
from sandbox.rocky.hrl.sparray import sparray
import numpy as np
import itertools


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

    def _get_flat_obs(self, hierarchical_obs):
        high_coord, low_coord = hierarchical_obs
        return high_coord * self.low_n_row * self.low_n_col + low_coord

    def step(self, action):
        next_obs, reward, done, info = self.flat_env.step(action)
        return Step(self._get_hierarchical_obs(next_obs), reward, done, **info)

    @property
    def analyzer(self):
        return HierarchicalGridWorldAnalyzer(self)


class HierarchicalGridWorldAnalyzer(object):
    def __init__(self, env):
        assert isinstance(env, HierarchicalGridWorldEnv)
        self.env = env
        self._posterior_sequences = dict()
        self._sequence_transition_probabilities = dict()

    def _wrap_black(self, str):
        return "\033[0;40;1;33m%s\033[m" % str

    def set_policy(self, policy):
        assert isinstance(policy, SubgoalPolicy)
        self.policy = policy

    def print_total_grid(self):
        for row in xrange(self.env.total_n_row):
            for col in xrange(self.env.total_n_col):
                high_row = row / self.env.low_n_row
                high_col = col / self.env.low_n_col
                if (high_row + high_col) % 2 == 0:
                    print(self._wrap_black(self.env.total_grid[row, col]), end="")
                else:
                    print(self.env.total_grid[row, col], end="")
            print()

    @property
    def n_states(self):
        return self.env.total_n_row * self.env.total_n_col

    def get_n_component_states(self, component_idx):
        if component_idx == 0:
            return self.env.high_n_row * self.env.high_n_col
        elif component_idx == 1:
            return self.env.low_n_row * self.env.low_n_col
        elif component_idx is None:
            return self.n_states
        else:
            raise NotImplementedError

    def get_obs_from_int_state(self, state):
        """
        Given a state in integer-based representation, return the actual observation which should be contained in the
        observation space of the environment
        :param state:
        :return:
        """
        return self.env._get_hierarchical_obs(state)

    def get_int_state_from_obs(self, obs):
        return self.env._get_flat_obs(obs)

    def get_int_component_state_from_obs(self, component_obs, component_idx):
        if component_idx is None:
            return self.get_int_state_from_obs(component_obs)
        # since it should already be an integer
        return component_obs

    def get_component_state(self, state, component_idx):
        """
        Given a state in integer-based representation and a component index, return the component state in
        integer-based representation
        :param state:
        :param component_idx:
        :return:
        """
        if component_idx is None:
            return state
        assert component_idx in [0, 1]
        return self.env._get_hierarchical_obs(state)[component_idx]

    @property
    def n_actions(self):
        return 4

    def compute_transition_probabilities(self):
        """
        Compute p(s'|s,a)
        :return:
        """
        # [0] -> state, [1] -> action, [2] -> next state
        probs = sparray((self.n_states, self.n_actions, self.n_states))
        # probs = np.zeros((self.n_states, self.n_actions, self.n_states))
        for state in xrange(self.n_states):
            for action in xrange(self.n_actions):
                possible_next_states = self.env.flat_env.get_possible_next_states(state, action)
                for next_state, prob in possible_next_states:
                    probs[state, action, next_state] += prob
        return probs

    def get_sequence_possible_next_states(self, state, action_sequence):
        """
        Compute a list of possible states sequences by following the action sequence, and their probabilities
        :param state: starting state
        :param action_sequence: sequence of actions
        :return:
        """
        if len(action_sequence) == 0:
            return [((state,), 1.)]
        first_action = action_sequence[0]
        rest_actions = action_sequence[1:]
        possible_next_states = self.env.flat_env.get_possible_next_states(state, first_action)
        all_possible_states = sparray((self.n_states,) * (len(action_sequence) + 1))
        for next_state, prob in possible_next_states:
            possible_further_states = self.get_sequence_possible_next_states(next_state, rest_actions)
            for further_states, further_prob in possible_further_states:
                all_possible_states[(state,) + further_states] += prob * further_prob
        return list(all_possible_states.iteritems())

    def compute_sequence_transition_probabilities(self, interval):
        """
        Compute p(s',{s},{a}|s) where {s} and {a} are
        :param interval:
        :return:
        """
        assert interval >= 1
        if interval in self._sequence_transition_probabilities:
            return self._sequence_transition_probabilities[interval]
        probs = sparray((self.n_states,) * (interval+1) + (self.n_actions,) * interval)
        for state in xrange(self.n_states):
            for action_sequence in itertools.product(*(xrange(self.n_actions) for _ in xrange(interval))):
                possible_next_states = self.get_sequence_possible_next_states(state, action_sequence)
                for next_states, prob in possible_next_states:
                    key = tuple(next_states) + tuple(action_sequence)
                    probs[key] += prob
        self._sequence_transition_probabilities[interval] = probs
        return probs

    def compute_posterior_sequences(self, interval):
        """
        Compute p({s},{a}|s,s'). This will be a dictionary with keys given by s and s'. The value will be a list of
        tuples (state_seq, action_seq, prob). It will hold that len(state_seq) = interval-1 and len(action_seq) =
        interval
        :param interval: interval of actions
        :return:
        """
        if interval in self._posterior_sequences:
            return self._posterior_sequences[interval]
        forward_probs = self.compute_sequence_transition_probabilities(interval)
        posteriors = dict()
        for key, prob in forward_probs.iteritems():
            state = key[0]
            next_state = key[interval]
            state_seq = key[1:interval]
            action_seq = key[interval+1:(interval * 2)+1]
            if (state, next_state) not in posteriors:
                posteriors[(state, next_state)] = list()
            posteriors[(state, next_state)].append((state_seq, action_seq, prob))
        norm_posteriors = dict()
        # normalize the probabilities
        for key, lst in posteriors.iteritems():
            norm_factor = np.sum((prob for _, _, prob in lst))
            norm_posteriors[key] = [
                (state_seq, action_seq, prob / norm_factor) for state_seq, action_seq, prob in lst
                ]
        self._posterior_sequences[interval] = norm_posteriors
        return norm_posteriors

    def get_posterior_sequences(self, interval, state, next_state):
        """
        :param policy:
        :param state:
        :param next_state:
        :return:
        """
        # assert isinstance(policy, SubgoalPolicy)
        # assert isinstance(policy.subgoal_space, Discrete)
        # interval = policy.subgoal_interval
        return self.compute_posterior_sequences(interval).get((state, next_state), list())

    def get_sequence_transition_probability(self, state, state_seq, action_seq):
        """
        Given s, {s}, {a}, compute p({s},{a}|s)
        :param state: starting state
        :param state_seq: sequence of states excluding the starting state
        :param action_seq: sequence of actions
        :return: compute the probability of the states given the starting state and the actions
        """
        assert len(state_seq) == len(action_seq)
        interval = len(state_seq)
        return self.compute_sequence_transition_probabilities(interval)[(state,) + tuple(state_seq) + tuple(action_seq)]

    def compute_goal_transition_probabilities(self, policy):
        """
        Compute p(s'|s,g) = sum_{{a},{s}} p(s',{a},{s}|s,g)
        :param policy: policy to compute the probabilities with
        :return:
        """
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, Discrete)
        n_goals = policy.subgoal_space.n
        interval = policy.subgoal_interval
        # [0] -> state, [1] -> goal, [2] -> next state
        probs = np.zeros((self.n_states, n_goals, self.n_states))
        for state in xrange(self.n_states):
            for next_state in xrange(self.n_states):
                for state_seq, action_seq, _ in self.get_posterior_sequences(interval, state, next_state):
                    s_prob = self.get_sequence_transition_probability(
                        state, state_seq + (next_state,), action_seq
                    )
                    prior_obs = map(self.env._get_hierarchical_obs, (state,) + state_seq)
                    flat_actions = np.asarray(map(self.env.action_space.flatten, action_seq))
                    for goal in xrange(n_goals):
                        flat_states = np.asarray(map(
                            policy.low_policy.observation_space.flatten,
                            [(x, goal) for x in prior_obs]
                        ))
                        a_dists = policy.low_policy.dist_info(flat_states, flat_actions)["prob"]
                        # select the actions we actually took
                        a_prob = np.prod([prob[a] for prob, a in zip(a_dists, action_seq)])
                        probs[state, goal, next_state] += s_prob * a_prob
        return probs
