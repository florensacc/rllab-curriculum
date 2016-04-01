import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc.special import to_onehot
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.spaces.discrete import Discrete


class StateGivenGoalMIEvaluator(LasagnePowered, Serializable):

    """
    Defines the reward bonus as the mutual information between the future state and the subgoal, given the current
    state: I(s',g|s) = H(s'|s) - H(s'|g,s) = E[log(p(s'|g,s)) - log(p(s'|s))]
    We train a neural network to fit a Gaussian to p(s'|g,s). Since we only have a discrete set of goals,
    we can marginalize over them to get p(s'|s):
    p(s'|s) = sum_g p(g|s)p(s'|g,s)
    """

    def __init__(
            self,
            env_spec,
            policy,
            regressor_cls=None,
            regressor_args=None,
            logger_delegate=None):
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, Discrete)
        assert isinstance(policy.high_policy.distribution, Categorical)
        assert isinstance(policy.low_policy.distribution, Categorical)

        Serializable.quick_init(self, locals())
        if regressor_cls is None:
            regressor_cls = GaussianMLPRegressor
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + policy.subgoal_space.flat_dim,),
            output_dim=env_spec.observation_space.flat_dim,
            name="(s'|g,s)",
            **regressor_args
        )
        self._subgoal_space = policy.subgoal_space
        self._logger_delegate = logger_delegate

    def _get_relevant_data(self, paths):
        obs = np.concatenate([p["observations"][:-1] for p in paths])
        next_obs = np.concatenate([p["observations"][1:] for p in paths])
        subgoals = np.concatenate([p["actions"][:-1] for p in paths])
        N = obs.shape[0]
        return obs.reshape((N, -1)), next_obs.reshape((N, -1)), subgoals

    def fit(self, paths):
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data(paths)
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        self._regressor.fit(xs, ys)
        if self._logger_delegate:
            self._logger_delegate.fit(paths)

    def log_diagnostics(self, paths):
        if self._logger_delegate:
            self._logger_delegate.log_diagnostics(paths)

    def predict(self, path):
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([path])
        N = flat_obs.shape[0]
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        high_probs = path["agent_infos"]["prob"]
        # high_pdists = path["pdists"]
        log_p_sprime_given_g_s = self._regressor.predict_log_likelihood(xs, ys)
        p_sprime_given_s = 0.
        n_subgoals = self._subgoal_space.n
        for goal in range(n_subgoals):
            goal_mat = np.tile(to_onehot(goal, n_subgoals), (N, 1))
            xs_goal = np.concatenate([flat_obs, goal_mat], axis=1)
            p_sprime_given_s += high_probs[:-1, goal] * np.exp(self._regressor.predict_log_likelihood(xs_goal, ys))
        ret = np.append(log_p_sprime_given_g_s - np.log(p_sprime_given_s + 1e-8), 0)
        if np.max(np.abs(ret)) > 1e3:
            import ipdb; ipdb.set_trace()
        return ret
