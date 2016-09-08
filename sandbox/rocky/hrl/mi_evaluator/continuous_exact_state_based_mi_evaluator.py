


import numpy as np

from rllab.misc import logger
from rllab.spaces.product import Product


class ContinuousExactStateBasedMIEvaluator(object):
    def __init__(self, env, policy, component_idx=None):
        # self.exact_computer = ExactComputer(env, policy, component_idx)
        self.env = env
        self.policy = policy
        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        self.component_idx = component_idx
        if component_idx is None:
            self.component_space = self.env.observation_space
        else:
            assert isinstance(self.env.observation_space, Product)
            self.component_space = self.env.observation_space.components[component_idx]
        self.computed = False

        self.p_next_state_given_goal_state = None
        self.p_next_state_given_state = None
        self.p_goal_given_state = None
        self.ent_next_state_given_goal_state = None
        self.ent_next_state_given_state = None
        self.mi_states = None
        self.mi_avg = None

    def get_p_sprime_given_s_g(self, state, goal, next_state):
        int_state = self.env.analyzer.get_int_state_from_obs(state)
        int_next_component_state = self.env.analyzer.get_int_component_state_from_obs(next_state, self.component_idx)

        # for next_state in xrange(self.env.analyzer.n_states):

        self.env.analyzer.get_posterior_sequences(self.subgoal_interval, int_state, next_state)
        import ipdb;
        ipdb.set_trace()
        return 0

    # def get_p_sprime_given_s(self, state, next_state):
    #     import ipdb;
    #     ipdb.set_trace()
    #     return 0

    def predict(self, path):
        raise NotImplementedError

    def update_cache(self):
        pass

    def log_diagnostics(self, paths):
        self.update_cache()
        analyzer = self.env.analyzer
        # we need to compute p(s'|s,g) = sum_{s_seq, a_seq} p(s',s_seq|s,a_seq) * p(a_seq|s_seq,s',g)
        # MI can be approximated by I(g,s'|s) = E_{g,s'} [log p(s'|g,s) - log p(s'|s)]
        mi_states = []
        n_goal_samples = 100#1000
        for state in range(analyzer.n_states):
            obs_state = analyzer.get_obs_from_int_state(state)
            # sample goals conditioned on this state
            goals, _ = self.policy.high_policy.get_actions([obs_state] * n_goal_samples)
            goals = self.policy.high_policy.action_space.flatten_n(goals)

            mi_state = 0.

            mi_goal_states = 0.

            p_csp_given_s_g = np.zeros((analyzer.get_n_component_states(self.component_idx), n_goal_samples))
            for next_state in range(analyzer.n_states):

                p_sp_given_s_g = 0.
                for state_seq, action_seq, prob in analyzer.get_posterior_sequences(self.subgoal_interval, state,
                                                                                    next_state):
                    ps = analyzer.get_sequence_transition_probability(state, state_seq + (next_state,), action_seq)
                    all_states = list(map(analyzer.get_obs_from_int_state, (state,) + state_seq))

                    flat_states = self.env.observation_space.flatten_n(all_states)
                    flat_actions = self.env.action_space.flatten_n(action_seq)
                    dup_flat_states = np.tile(np.expand_dims(flat_states, axis=1), (1, n_goal_samples, 1))
                    dup_goals = np.tile(np.expand_dims(goals, axis=0), (self.subgoal_interval, 1, 1))
                    dup_actions = np.tile(np.expand_dims(flat_actions, axis=1), (1, n_goal_samples, 1))

                    reshape_dup_flat_states = dup_flat_states.reshape((-1, dup_flat_states.shape[-1]))
                    reshape_dup_goals = dup_goals.reshape((-1, dup_goals.shape[-1]))
                    reshape_dup_actions = dup_actions.reshape((-1, dup_actions.shape[-1]))

                    reshape_dup_concat = np.concatenate([reshape_dup_flat_states, reshape_dup_goals], axis=-1)
                    a_probs = np.exp(self.policy.low_policy.distribution.log_likelihood(
                        reshape_dup_actions,
                        self.policy.low_policy.dist_info(reshape_dup_concat, dict())
                    )).reshape((self.subgoal_interval, n_goal_samples)).prod(axis=0)

                    p_sp_given_s_g += ps * a_probs
                cmp_state = analyzer.get_component_state(next_state, self.component_idx)
                p_csp_given_s_g[cmp_state] += p_sp_given_s_g
            p_csp_given_s = np.mean(p_csp_given_s_g, axis=-1, keepdims=True)
            mi_goal_states += np.mean(
                np.sum(
                    p_csp_given_s_g * np.log(p_csp_given_s_g + 1e-8) - np.log(p_csp_given_s + 1e-8) * p_csp_given_s,
                    axis=0
                )
            )
            mi_states.append(mi_goal_states)
        mi_avg = np.mean(mi_states)
        logger.record_tabular("Approx.I(goal,next_state|state)", mi_avg)

    def fit(self, paths):
        pass
