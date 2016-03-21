from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.mdp.subgoal_mdp import SubgoalMDP, SubgoalMDPSpec
from rllab.regressor.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.misc.special import to_onehot
import numpy as np


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
            mdp_spec,
            regressor_cls=None,
            regressor_args=None):
        assert isinstance(mdp_spec, SubgoalMDPSpec)

        Serializable.quick_init(self, locals())
        if regressor_cls is None:
            regressor_cls = GaussianMLPRegressor
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = regressor_cls(
            input_shape=(mdp_spec.observation_dim + mdp_spec.n_subgoals,),
            output_dim=mdp_spec.observation_dim,
            name="(s'|g,s)",
            **regressor_args
        )
        self._n_subgoals = mdp_spec.n_subgoals

    def _get_relevant_data(self, paths):
        obs = np.concatenate([p["observations"][:-1] for p in paths])
        next_obs = np.concatenate([p["observations"][1:] for p in paths])
        subgoals = np.concatenate([p["subgoals"][:-1] for p in paths])
        N = obs.shape[0]
        return obs.reshape((N, -1)), next_obs.reshape((N, -1)), subgoals

    def fit(self, paths):
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data(paths)
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        self._regressor.fit(xs, ys)

    def predict(self, path):
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([path])
        N = flat_obs.shape[0]
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        high_pdists = path["high_pdists"]
        log_p_sprime_given_g_s = self._regressor.predict_log_likelihood(xs, ys)
        p_sprime_given_s = 0.
        for goal in range(self._n_subgoals):
            goal_mat = np.tile(to_onehot(goal, self._n_subgoals), (N, 1))
            xs_goal = np.concatenate([flat_obs, goal_mat], axis=1)
            p_sprime_given_s += high_pdists[:-1, goal] * np.exp(self._regressor.predict_log_likelihood(xs_goal, ys))

        ret = np.append(log_p_sprime_given_g_s - np.log(p_sprime_given_s), 0)
        if np.any(np.isinf(ret)):
            import ipdb; ipdb.set_trace()
        return np.append(log_p_sprime_given_g_s - np.log(p_sprime_given_s), 0)
