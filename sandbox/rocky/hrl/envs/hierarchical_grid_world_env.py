from __future__ import absolute_import
from __future__ import print_function

import itertools
import sys

import joblib
import numpy as np
import theano
import theano.tensor as TT

from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.misc import logger
from rllab.sampler.utils import rollout
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from sandbox.rocky.hrl.misc.sparray import sparray
from sandbox.rocky.hrl.policies.subgoal_policy import SubgoalPolicy, FixedGoalPolicy


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
        self.analyzer = HierarchicalGridWorldAnalyzer(self)

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
        high_row = high_coord / self.high_n_col
        high_col = high_coord % self.high_n_col
        low_row = low_coord / self.low_n_col
        low_col = low_coord % self.low_n_col
        total_row = high_row * self.low_n_row + low_row
        total_col = high_col * self.low_n_col + low_col
        return total_row * self.total_n_col + total_col

    def step(self, action):
        next_obs, reward, done, info = self.flat_env.step(action)
        return Step(self._get_hierarchical_obs(next_obs), reward, done, **info)

    def render(self):
        self.analyzer.print_total_grid()
        print("")


class SymSharedArrs(object):
    def __init__(self):
        self.states = theano.shared(np.zeros((1,), dtype='int'), 'lookup_states')
        self.next_component_states = theano.shared(np.zeros((1,), dtype='int'), 'lookup_next_component_states')
        self.terminal_ids = theano.shared(np.zeros((1,), dtype='int'), 'terminal_ids')
        self.goals = theano.shared(np.zeros((1,), dtype='int'), 'lookup_goals')


class HierarchicalGridWorldAnalyzer(object):
    def __init__(self, env):
        assert isinstance(env, HierarchicalGridWorldEnv)
        self.env = env
        self.policy = None
        self._posterior_sequences = dict()
        self._sequence_transition_probabilities = dict()
        self._sym_shared_arrs = None

    @classmethod
    def from_pkl(cls, file_name):
        params = joblib.load(file_name)
        env = params["env"]
        policy = params["policy"]
        analyzer = HierarchicalGridWorldAnalyzer(env)
        analyzer.set_policy(policy)
        return analyzer

    def _wrap_black(self, str):
        return "\033[0;40;1;33m%s\033[m" % str

    def _wrap_current(self, str):
        return "\033[0;43;1;30m%s\033[m" % str

    def set_policy(self, policy):
        assert isinstance(policy, SubgoalPolicy)
        self.policy = policy

    def print_total_grid(self, file=sys.stdout):
        current = self.env.flat_env.state
        current_row = current / self.env.total_n_col
        current_col = current % self.env.total_n_col
        for row in xrange(self.env.total_n_row):
            for col in xrange(self.env.total_n_col):
                high_row = row / self.env.low_n_row
                high_col = col / self.env.low_n_col
                if row == current_row and col == current_col:
                    print(self._wrap_current(self.env.total_grid[row, col]), end="", file=file)
                elif (high_row + high_col) % 2 == 0:
                    print(self._wrap_black(self.env.total_grid[row, col]), end="", file=file)
                else:
                    print(self.env.total_grid[row, col], end="", file=file)
            print(file=file)

    def render_rollout(self):
        path = rollout(self.env, self.policy, max_path_length=100)
        states = map(self.get_int_state_from_obs, map(self.env.observation_space.unflatten, path["observations"]))
        for state in states:
            self.env.flat_env.state = state
            self.print_total_grid()
            print("")

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
        probs = sparray((self.n_states,) * (interval + 1) + (self.n_actions,) * interval)
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
        logger.log("computing posterior_sequences")
        forward_probs = self.compute_sequence_transition_probabilities(interval)
        posteriors = dict()
        for key, prob in forward_probs.iteritems():
            state = key[0]
            next_state = key[interval]
            state_seq = key[1:interval]
            action_seq = key[interval + 1:(interval * 2) + 1]
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

    # def get_max_number_posterior_sequences(self, interval):
    #     for state in xrange(self.n_states):
    #         pass

    def compute_goal_transition_probabilities(self):
        """
        Compute p(s'|s,g) = sum_{{a},{s}} p(s',{a},{s}|s,g)
        :return:
        """
        logger.log("computing goal_transition_probabilities")
        policy = self.policy
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
                        a_dists = policy.low_policy.dist_info(flat_states, dict())["prob"]
                        # select the actions we actually took
                        a_prob = np.prod([prob[a] for prob, a in zip(a_dists, action_seq)])
                        probs[state, goal, next_state] += s_prob * a_prob
        return probs

    def print_state_visitation_frequency(self, max_path_length=100, n_paths=50):
        paths = []
        for _ in xrange(n_paths):
            paths.append(rollout(env=self.env, agent=self.policy, max_path_length=max_path_length))
        observations = np.vstack([p["observations"] for p in paths])
        self.print_total_frequency(observations)
        self.print_high_frequency(observations)

    def print_state_visitation_frequency_per_goal(self, max_path_length=100, n_paths=50):
        for goal in xrange(self.policy.subgoal_space.n):
            print("Goal #%d" % (goal + 1))
            paths = []
            for _ in xrange(n_paths):
                paths.append(
                    rollout(env=self.env, agent=FixedGoalPolicy(self.policy, goal), max_path_length=max_path_length))
            observations = np.vstack([p["observations"] for p in paths])
            self.print_total_frequency(observations)
            self.print_high_frequency(observations)

    def print_total_frequency(self, observations):
        total_obs = map(self.env.observation_space.unflatten, observations)
        total_obs = map(self.get_int_state_from_obs, total_obs)
        total_onehots = map(self.env.flat_env.observation_space.flatten, total_obs)
        mean_onehots = np.mean(total_onehots, axis=0).reshape(
            (self.env.total_n_row, self.env.total_n_col)
        )
        print(np.array2string(mean_onehots, formatter={'float_kind': lambda x: "%.2f" % x}))

    def print_high_frequency(self, observations):
        component_space = self.env.observation_space.components[0]
        high_states = np.array(
            [component_space.flatten(self.env.observation_space.unflatten(x)[0]) for x in observations])
        mean_high_states = np.mean(high_states, axis=0).reshape(
            (self.env.high_n_row, self.env.high_n_col)
        )
        print(np.array2string(mean_high_states, formatter={'float_kind': lambda x: "%.2f" % x}))

    def prepare_sym(self, paths, component_idx):
        if self._sym_shared_arrs is None:
            return
        states_shared = self._sym_shared_arrs.states
        next_component_states_shared = self._sym_shared_arrs.next_component_states
        goals_shared = self._sym_shared_arrs.goals
        terminal_ids_shared = self._sym_shared_arrs.terminal_ids

        new_states = []
        new_next_component_states = []
        new_goals = []
        new_terminal_ids = []

        interval = self.policy.subgoal_interval

        for path in paths:
            path_obs = path["observations"]
            path_goals = path["agent_infos"]["subgoal"]

            path_length = len(path_obs)

            unflat_obs = map(self.env.observation_space.unflatten, path_obs)
            int_obs = map(self.get_int_state_from_obs, unflat_obs)
            sub_int_obs = int_obs[::interval]
            component_obs = [self.get_int_component_state_from_obs(self.get_component_state(x, component_idx),
                                                                   component_idx) for x in int_obs]
            sub_component_obs = component_obs[::interval]
            next_sub_component_obs = np.append(sub_component_obs[1:], 0)

            int_goals = map(self.policy.subgoal_space.unflatten, path_goals)
            sub_int_goals = int_goals[::interval]

            int_obs = np.repeat(sub_int_obs, interval, axis=0)[:path_length]
            next_component_obs = np.repeat(next_sub_component_obs, interval, axis=0)[:path_length]
            int_goals = np.repeat(sub_int_goals, interval, axis=0)[:path_length]

            new_states.extend(int_obs)
            new_next_component_states.extend(next_component_obs)
            new_terminal_ids.append(len(new_next_component_states) - 1)
            new_goals.extend(int_goals)

        states_shared.set_value(np.asarray(new_states))
        next_component_states_shared.set_value(np.asarray(new_next_component_states))
        goals_shared.set_value(np.asarray(new_goals))
        terminal_ids_shared.set_value(np.asarray(new_terminal_ids))

    def mi_bonus_sym(self, component_idx):
        if self._sym_shared_arrs is None:
            self._sym_shared_arrs = SymSharedArrs()
            # initialize shared arrays
            # these should be of the same length as the number of low-level states, actions, etc.
            # basically, what we need to do

        states_shared = self._sym_shared_arrs.states
        next_component_states_shared = self._sym_shared_arrs.next_component_states
        terminal_ids = self._sym_shared_arrs.terminal_ids
        goals_shared = self._sym_shared_arrs.goals

        lookup_sym = self.compute_mi_bonus_lookup_sym(component_idx)
        bonus_sym = lookup_sym[states_shared, next_component_states_shared, goals_shared]
        bonus_sym = TT.set_subtensor(bonus_sym[terminal_ids], 0)
        return bonus_sym

    def compute_mi_bonus_lookup_sym(self, component_idx):
        """
        Compute information needed by symbolic computation
        :return:
        """

        ###
        # the symbolic representation of the reward is actually annoyingly difficult to compute.
        #
        # The exact formula is log p(s'|s,g) - log p(s'|s)
        #
        # First we expand the first term as a sum over all possible intermediate state and action sequences:
        # p(s'|s,g) = sum_{s_seq, a_seq} p(s_seq, s'|s, a_seq) * p(a_seq|s_seq, s, g)
        # similarly the second term can be obtained by p(s'|s) = sum_g p(s'|s,g)p(g|s)
        #
        # What's annoying is that we'd need to vectorize this whole thing...
        #
        # To start, we'd need to maintain some high-D tensors:
        # list of states: SequenceId * TimeStepId -> StateSeqEntry[Int]
        states_arr = []
        # list of actions: SequenceId * TimeStepId -> ActionSeqEntry[Int]
        actions_arr = []
        # list of p(s_seq,s'|s,a_seq): SequenceId -> Prob[Float]
        tprob_arr = []
        # mapping from sequence id to state index: SequenceId -> StateId[Int]
        state_ids = []
        # mapping from sequence id to next component state index: SequenceId -> NextComponentStateId[Int]
        next_component_state_ids = []

        n_states = self.n_states
        n_component_states = self.get_n_component_states(component_idx)
        n_subgoals = self.policy.subgoal_space.n
        interval = self.policy.subgoal_interval
        # Now let's fill in these arrays
        for state in xrange(n_states):
            for next_state in xrange(n_states):
                next_component_state = self.get_component_state(next_state, component_idx)
                for state_seq, action_seq, _ in self.get_posterior_sequences(interval, state, next_state):
                    prob = self.get_sequence_transition_probability(state, state_seq + (next_state,),
                                                                    action_seq)
                    states_arr.append((state,) + state_seq)
                    actions_arr.append(action_seq)
                    tprob_arr.append(prob)
                    state_ids.append(state)
                    next_component_state_ids.append(next_component_state)

        flat_states = np.array(
            [[self.env.observation_space.flatten(self.get_obs_from_int_state(x)) for x in xs] for xs in
             states_arr], dtype='uint8'
        )
        flat_actions = np.array(
            [[self.env.action_space.flatten(x) for x in xs] for xs in actions_arr], dtype='uint8'
        )
        flat_tprobs = np.array(tprob_arr)
        flat_states_shared = theano.shared(flat_states)
        flat_actions_shared = theano.shared(flat_actions)
        flat_tprobs_shared = theano.shared(flat_tprobs)

        p_sp_given_s_g_sym = TT.zeros((n_states, n_component_states, n_subgoals))
        p_sp_given_s_sym = TT.zeros((n_states, n_component_states))
        # p_g_given_s_sym = TT.zeros((n_states, n_subgoals))

        nonseq_states = []
        for state in xrange(n_states):
            nonseq_states.append(self.env.observation_space.flatten(self.get_obs_from_int_state(state)))
        nonseq_states = np.asarray(nonseq_states, dtype='uint8')
        nonseq_states_shared = theano.shared(nonseq_states)

        high_dist_info_sym = self.policy.high_policy.dist_info_sym(nonseq_states_shared, dict())

        for goal in xrange(n_subgoals):
            subgoals = TT.zeros((flat_states_shared.shape[0], flat_states_shared.shape[1], n_subgoals))
            subgoals = TT.set_subtensor(subgoals[:, :, goal], 1)
            flat_states_with_subgoal = TT.concatenate([flat_states_shared, subgoals], axis=2)
            state_dim = flat_states_with_subgoal.shape[-1]
            action_dim = flat_actions_shared.shape[-1]

            nonseq_subgoal_sym = TT.zeros((n_states, n_subgoals), dtype='uint8')
            nonseq_subgoal_sym = TT.set_subtensor(nonseq_subgoal_sym[:, goal], 1)

            states_2d = flat_states_with_subgoal.reshape((-1, state_dim))
            actions_2d = flat_actions_shared.reshape((-1, action_dim))
            dist_info_sym = self.policy.low_policy.dist_info_sym(states_2d, dict())
            action_prob_2d_sym = TT.exp(
                self.policy.low_policy.distribution.log_likelihood_sym(actions_2d, dist_info_sym))
            action_prob_sym = action_prob_2d_sym.reshape((-1, interval))
            seq_prob = flat_tprobs_shared * TT.prod(action_prob_sym, axis=-1)
            # a weird way to aggregate the result, taking advantage of the fact that inc_subtensor allow
            # repeating indices
            p_sp_given_s_g_sym = TT.inc_subtensor(p_sp_given_s_g_sym[state_ids, next_component_state_ids, goal],
                                                  seq_prob)
            p_g_given_s_sym = TT.exp(
                self.policy.high_policy.distribution.log_likelihood_sym(nonseq_subgoal_sym, high_dist_info_sym)
            )
            p_sp_given_s_sym = p_sp_given_s_sym + p_sp_given_s_g_sym[:, :, goal] * p_g_given_s_sym.dimshuffle(0, 'x')

        # for goal in xrange(n_subgoals):
        #     p_sp_g_given_s_sym = p_sp_given_s_g_sym[:,:,goal] *
        #     p_sp_given_s_sym = TT.inc_subtensor()
        # p_sp_given_s_sym = TT.sum(p_sp_given_s_g_sym, axis=-1, keepdims=True)
        mi_bonus_sym = TT.log(p_sp_given_s_g_sym + 1e-8) - TT.log(p_sp_given_s_sym + 1e-8).dimshuffle(0, 1, 'x')

        return mi_bonus_sym
