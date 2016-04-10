from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from rllab.envs.grid_world_env import GridWorldEnv
from rllab.spaces.discrete import Discrete
from sandbox.rocky.hrl.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy


class ExactComputer(object):
    def __init__(self, env, policy):
        assert isinstance(env, CompoundActionSequenceEnv)
        assert isinstance(env.wrapped_env, GridWorldEnv)
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, Discrete)
        assert env._reset_history
        self.env = env
        self.policy = policy
        self.n_states = env.wrapped_env.observation_space.n
        self.n_raw_actions = env.wrapped_env.action_space.n
        self.n_subgoals = policy.subgoal_space.n
        self.subgoal_interval = policy.subgoal_interval
        self._computed = False

    def compute_p_goal_given_state(self):
        p_goal_given_state = np.zeros((self.n_states, self.n_subgoals))
        for state in xrange(self.n_states):
            # index: [0] -> goal
            p_goal_given_state[state] = np.copy(self.policy.high_policy.get_action(state)[1]["prob"])
        return p_goal_given_state

    def compute_p_next_state_given_goal_state(self):
        # index: [0] -> goal, [1] -> state, [2] -> next state
        p_next_state_given_goal_state = np.zeros((self.n_subgoals, self.n_states, self.n_states))
        for state in xrange(self.n_states):
            for raw_action in xrange(self.n_raw_actions):
                self.env.wrapped_env.set_state(state)
                next_state = self.env.wrapped_env.step(raw_action).observation
                if next_state != state:
                    action_sequence = self.env._action_map[raw_action]
                    # We need to compute the probability of generating the raw action, given a certain subgoal
                    for subgoal in xrange(self.n_subgoals):
                        seq_prob = 1.
                        for timestep, action in enumerate(action_sequence):
                            seq_prob *= self.policy.low_policy.get_action((state, subgoal, timestep))[1]["prob"][action]
                        p_next_state_given_goal_state[subgoal, state, next_state] += seq_prob
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                p_next_state_given_goal_state[subgoal, state, state] = max(0., 1. - np.sum(
                    p_next_state_given_goal_state[subgoal, state, :]))
        return p_next_state_given_goal_state

    def compute_p_next_state_given_state(self, p_next_state_given_goal_state, p_goal_given_state):
        p_next_state_given_state = np.zeros((self.n_states, self.n_states))
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_states):
                    p_next_state_given_state[state, next_state] += \
                        p_next_state_given_goal_state[subgoal, state, next_state] * p_goal_given_state[state, subgoal]
        return p_next_state_given_state

    def compute_ent_next_state_given_state(self, p_next_state_given_state):
        # index: [0] -> state, [1] -> next state
        ent_next_state_given_state = np.zeros((self.n_states,))
        for state in xrange(self.n_states):
            for next_state in xrange(self.n_states):
                ent_next_state_given_state[state] += -p_next_state_given_state[state, next_state] * np.log(
                    p_next_state_given_state[state, next_state] + 1e-8)
        return ent_next_state_given_state

    def compute_ent_next_state_given_goal_state(self, p_next_state_given_goal_state):
        # index: [0] -> goal, [1] -> state, [2] -> next state
        ent_next_state_given_goal_state = np.zeros((self.n_subgoals, self.n_states))
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_states):
                    ent_next_state_given_goal_state[subgoal, state] += \
                        -p_next_state_given_goal_state[subgoal, state, next_state] * np.log(
                            p_next_state_given_goal_state[subgoal, state, next_state] + 1e-8)
        return ent_next_state_given_goal_state

    def compute_mi_states(self, ent_next_state_given_state, ent_next_state_given_goal_state, p_goal_given_state):
        mi_states = np.zeros((self.n_states,))
        for state in xrange(self.n_states):
            mi = ent_next_state_given_state[state]
            for subgoal in xrange(self.n_subgoals):
                mi -= ent_next_state_given_goal_state[subgoal, state] * p_goal_given_state[state, subgoal]
            mi_states[state] = mi
        return mi_states
