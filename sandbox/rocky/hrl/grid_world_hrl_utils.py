from __future__ import absolute_import
from __future__ import print_function

import numpy as np

from rllab.envs.grid_world_env import GridWorldEnv
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy


class ExactComputer(object):
    def __init__(self, env, policy, component_idx=None):
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, Discrete)
        self.env = env
        self.analyzer = analyzer = env.analyzer
        self.policy = policy
        self.component_idx = component_idx
        if component_idx is not None:
            assert isinstance(self.env.observation_space, Product)
        self.n_states = analyzer.n_states
        self.n_component_states = analyzer.get_n_component_states(component_idx)
        self.n_subgoals = policy.subgoal_space.n
        self.subgoal_interval = policy.subgoal_interval

    def compute_p_goal_given_state(self):
        logger.log("computing p_goal_given_state")
        p_goal_given_state = np.zeros((self.n_states, self.n_subgoals))
        for state in xrange(self.n_states):
            obs = self.analyzer.get_obs_from_int_state(state)
            # index: [0] -> goal
            p_goal_given_state[state] = np.copy(self.policy.high_policy.get_action(obs)[1]["prob"])
        return p_goal_given_state

    def compute_p_next_state_given_goal_state(self):
        logger.log("computing p_next_state_given_goal_state")
        # [0] -> state, [1] -> goal, [2] -> next state
        goal_transition_probs = self.analyzer.compute_goal_transition_probabilities(self.policy)
        # return value index: [0] -> goal, [1] -> state, [2] -> next state
        p_next_state_given_goal_state = np.zeros((self.n_subgoals, self.n_states, self.n_component_states))
        for state in xrange(self.n_states):
            for goal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_states):
                    component_state = self.analyzer.get_component_state(next_state, self.component_idx)
                    p_next_state_given_goal_state[goal, state, component_state] += \
                        goal_transition_probs[state, goal, next_state]
        return p_next_state_given_goal_state

    def compute_p_next_state_given_state(self, p_next_state_given_goal_state, p_goal_given_state):
        logger.log("computing p_next_state_given_state")
        # index: [0] -> state, [1] -> next state
        p_next_state_given_state = np.zeros((self.n_states, self.n_component_states))
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_component_states):
                    p_next_state_given_state[state, next_state] += \
                        p_next_state_given_goal_state[subgoal, state, next_state] * p_goal_given_state[state, subgoal]
        return p_next_state_given_state

    def compute_ent_next_state_given_state(self, p_next_state_given_state):
        logger.log("computing ent_next_state_given_state")
        # index: [0] -> state, [1] -> next state
        ent_next_state_given_state = np.zeros((self.n_states,))
        for state in xrange(self.n_states):
            for next_state in xrange(self.n_component_states):
                ent_next_state_given_state[state] += -p_next_state_given_state[state, next_state] * np.log(
                    p_next_state_given_state[state, next_state] + 1e-8)
        return ent_next_state_given_state

    def compute_ent_next_state_given_goal_state(self, p_next_state_given_goal_state):
        logger.log("computing ent_next_state_given_goal_state")
        # index: [0] -> goal, [1] -> state
        ent_next_state_given_goal_state = np.zeros((self.n_subgoals, self.n_states))
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_component_states):
                    ent_next_state_given_goal_state[subgoal, state] += \
                        -p_next_state_given_goal_state[subgoal, state, next_state] * np.log(
                            p_next_state_given_goal_state[subgoal, state, next_state] + 1e-8)
        return ent_next_state_given_goal_state

    def compute_mi_states(self, ent_next_state_given_state, ent_next_state_given_goal_state, p_goal_given_state):
        logger.log("computing mi_states")
        mi_states = np.zeros((self.n_states,))
        for state in xrange(self.n_states):
            mi = ent_next_state_given_state[state]
            for subgoal in xrange(self.n_subgoals):
                mi -= ent_next_state_given_goal_state[subgoal, state] * p_goal_given_state[state, subgoal]
            mi_states[state] = mi
        return mi_states
