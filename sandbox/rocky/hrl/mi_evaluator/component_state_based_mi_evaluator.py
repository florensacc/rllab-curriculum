from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc import special
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product


class ComponentStateBasedMIEvaluator(LasagnePowered, Serializable):

    def __init__(
            self,
            env_spec,
            policy,
            component_idx,
            regressor_cls=None,
            regressor_args=None,
            logger_delegate=None):
        """
        Compute the bonus reward as given by I(g,s'_i|s), where i is given by the provided component index.
        :param env_spec: Spec for the environment
        :param policy: Hierarchical policy
        :param component_idx: Index for the component in the observation that we care about
        :param regressor_cls: Class of the regressor, which defaults to GaussianMLPRegressor
        :param regressor_args: Extra constructor arguments of the regressor
        :return:
        """
        assert isinstance(env_spec.observation_space, Product)
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, Discrete)
        assert isinstance(policy.high_policy.distribution, Categorical)
        assert isinstance(policy.low_policy.distribution, Categorical)
        assert 0 <= component_idx < len(env_spec.observation_space.components)

        Serializable.quick_init(self, locals())
        if regressor_cls is None:
            regressor_cls = GaussianMLPRegressor
        if regressor_args is None:
            regressor_args = dict()

        self.env_spec = env_spec
        self.component_idx = component_idx
        self.component_space = env_spec.observation_space.components[component_idx]
        self.regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + policy.subgoal_space.flat_dim,),
            output_dim=self.component_space.flat_dim,
            name="(s'|g,s)",
            **regressor_args
        )
        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        self.logger_delegate = logger_delegate

    def _get_relevant_data(self, paths):
        obs = np.concatenate([p["agent_infos"]["high_obs"][:-1] for p in paths])
        N = obs.shape[0]
        next_obs = np.concatenate([p["agent_infos"]["high_obs"][1:] for p in paths]).reshape((N, -1))

        obs_flat_dims = [c.flat_dim for c in self.env_spec.observation_space.components]
        slice_start = sum(obs_flat_dims[:self.component_idx])
        slice_end = slice_start + obs_flat_dims[self.component_idx]
        next_component_obs = next_obs[:, slice_start:slice_end]
        subgoals = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
        return obs.reshape((N, -1)), next_component_obs, subgoals

    def fit(self, paths):
        subsampled_paths = paths#[hrl_utils.subsample_path(p, self.subgoal_interval) for p in paths]
        flat_obs, flat_next_component_obs, subgoals = self._get_relevant_data(subsampled_paths)
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_component_obs
        self.regressor.fit(xs, ys)
        if self.logger_delegate:
            self.logger_delegate.fit(paths)

    def predict(self, path):
        path_length = len(path["rewards"])
        #path = hrl_utils.subsample_path(path, self.subgoal_interval)
        flat_obs, flat_next_component_obs, subgoals = self._get_relevant_data([path])
        N = flat_obs.shape[0]
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_component_obs
        high_probs = path["agent_infos"]["high"]["prob"]
        log_p_sprime_given_g_s = self.regressor.predict_log_likelihood(xs, ys)
        p_sprime_given_s = 0.
        n_subgoals = self.subgoal_space.n
        for goal in range(n_subgoals):
            goal_mat = np.tile(special.to_onehot(goal, n_subgoals), (N, 1))
            xs_goal = np.concatenate([flat_obs, goal_mat], axis=1)
            p_sprime_given_s += high_probs[:-1, goal] * np.exp(self.regressor.predict_log_likelihood(xs_goal, ys))
        bonuses = np.append(log_p_sprime_given_g_s - np.log(p_sprime_given_s + 1e-8), 0)
        # bonuses = np.tile(
        #     np.expand_dims(bonuses, axis=1),
        #     (1, self.subgoal_interval)
        # ).flatten()[:path_length]
        return bonuses

    def log_diagnostics(self, paths):
        if self.logger_delegate:
            self.logger_delegate.log_diagnostics(paths)

    # def get_predicted_mi(self, env, policy):
    #     exact_computer = ExactComputer(env, policy)
    #     # We need to compute p(s'|g,s) approximately
    #
    #     n_subgoals = policy.subgoal_space.n
    #     n_states = env.observation_space.n
    #     # index: [0] -> goal, [1] -> state, [2] -> next state
    #     p_next_state_given_goal_state = np.zeros((n_subgoals, n_states, n_states))
    #
    #     for state in xrange(n_states):
    #         for subgoal in xrange(n_subgoals):
    #             for next_state in xrange(n_states):
    #                 xs = [
    #                     np.concatenate([env.observation_space.flatten(state), policy.subgoal_space.flatten(subgoal)])
    #                 ]
    #                 ys = [
    #                     env.observation_space.flatten(next_state)
    #                 ]
    #                 p_next_state_given_goal_state[subgoal, state, next_state] = np.exp(
    #                     self.regressor.predict_log_likelihood(xs, ys)[0])
    #
    #     p_goal_given_state = exact_computer.compute_p_goal_given_state()
    #     p_next_state_given_state = exact_computer.compute_p_next_state_given_state(
    #         p_next_state_given_goal_state=p_next_state_given_goal_state,
    #         p_goal_given_state=p_goal_given_state
    #     )
    #
    #     # Now we can compute the entropies
    #     ent_next_state_given_state = exact_computer.compute_ent_next_state_given_state(p_next_state_given_state)
    #     ent_next_state_given_goal_state = exact_computer.compute_ent_next_state_given_goal_state(
    #         p_next_state_given_goal_state)
    #     mi_states = exact_computer.compute_mi_states(
    #         ent_next_state_given_goal_state=ent_next_state_given_goal_state,
    #         ent_next_state_given_state=ent_next_state_given_state,
    #         p_goal_given_state=p_goal_given_state
    #     )
    #     return np.mean(mi_states)
