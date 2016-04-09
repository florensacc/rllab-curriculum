from rllab.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.spaces.discrete import Discrete
from rllab.misc import logger
import numpy as np


class ExactStateGivenGoalMIEvaluator2(object):
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

    def _get_relevant_data(self, paths):
        obs = np.concatenate([p["observations"][:-1] for p in paths])
        next_obs = np.concatenate([p["observations"][1:] for p in paths])
        subgoals = np.concatenate([p["actions"][:-1] for p in paths])
        N = obs.shape[0]
        return obs.reshape((N, -1)), next_obs.reshape((N, -1)), subgoals

    def predict(self, path):
        self._update_cache()
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([path])
        obs = [self.env.observation_space.unflatten(o) for o in flat_obs]
        next_obs = [self.env.observation_space.unflatten(o) for o in flat_next_obs]
        subgoals = [self.policy.subgoal_space.unflatten(g) for g in subgoals]
        ret = np.log(self._p_next_state_given_goal_state[subgoals, obs, next_obs] + 1e-8) - np.log(
            self._p_next_state_given_state[obs, next_obs] + 1e-8)
        return np.append(ret, 0)

    def _update_cache(self):
        # We need to compute the quantity I(g, s'|s) = H(s'|s) - H(s'|g,s)
        # To compute this, we need to compute p(s'|g,s) and p(s'|s), which in turn needs p(g|s)
        # The second one is simply given by the policy. For the first one we need to do some work
        # We have p(s'|g,s) = sum_a p(s'|a,s) p(a|g,s)

        # We should only recompute when the policy changes

        if self._computed:
            return
        self._computed = True
        # index: [0] -> goal, [1] -> state, [2] -> next state
        p_next_state_given_goal_state = np.zeros((self.n_subgoals, self.n_states, self.n_states))

        # index: [0] -> state, [1] -> next state
        p_next_state_given_state = np.zeros((self.n_states, self.n_states))

        # index: [0] -> state, [1] -> goal
        p_goal_given_state = np.zeros((self.n_states, self.n_subgoals))

        for state in xrange(self.n_states):
            # index: [0] -> goal
            p_goal_given_state[state] = self.policy.high_policy.get_action(state)[1]["prob"]
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
                p_next_state_given_goal_state[subgoal, state, state] = 1. - np.sum(p_next_state_given_goal_state[
                                                                                   subgoal, state, :])
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_states):
                    p_next_state_given_state[state, next_state] += \
                        p_next_state_given_goal_state[subgoal, state, next_state] * p_goal_given_state[state, subgoal]
        # Now we can compute the entropies
        # index: [0] -> state, [1] -> next state
        ent_next_state_given_state = np.zeros((self.n_states,))
        for state in xrange(self.n_states):
            for next_state in xrange(self.n_states):
                ent_next_state_given_state[state] += -p_next_state_given_state[state, next_state] * np.log(
                    p_next_state_given_state[state, next_state] + 1e-8)
        # index: [0] -> goal, [1] -> state, [2] -> next state
        ent_next_state_given_goal_state = np.zeros((self.n_subgoals, self.n_states))
        for state in xrange(self.n_states):
            for subgoal in xrange(self.n_subgoals):
                for next_state in xrange(self.n_states):
                    ent_next_state_given_goal_state[subgoal, state] += \
                        -p_next_state_given_goal_state[subgoal, state, next_state] * np.log(
                            p_next_state_given_goal_state[subgoal, state, next_state] + 1e-8)
        mi_states = np.zeros((self.n_states,))
        for state in xrange(self.n_states):
            mi = ent_next_state_given_state[state]
            for subgoal in xrange(self.n_subgoals):
                mi -= ent_next_state_given_goal_state[subgoal, state] * p_goal_given_state[state, subgoal]
            mi_states[state] = mi

        self._p_next_state_given_goal_state = p_next_state_given_goal_state
        self._p_next_state_given_state = p_next_state_given_state
        self._p_goal_given_state = p_goal_given_state
        self._ent_next_state_given_goal_state = ent_next_state_given_goal_state
        self._ent_next_state_given_state = ent_next_state_given_state
        self._mi_states = mi_states
        self._mi_avg = np.mean(mi_states)

    def log_diagnostics(self, paths):
        self._update_cache()
        logger.record_tabular("I(goal,next_state|state)", self._mi_avg)

    def fit(self, paths):
        # calling fit = invalidate caches
        self._computed = False

