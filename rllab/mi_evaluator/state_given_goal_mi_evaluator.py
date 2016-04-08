import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc.special import to_onehot
from rllab.misc import tensor_utils
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

        self.regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + policy.subgoal_space.flat_dim,),
            output_dim=env_spec.observation_space.flat_dim,
            name="(s'|g,s)",
            **regressor_args
        )
        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        self.logger_delegate = logger_delegate

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
        self.regressor.fit(xs, ys)
        if self.logger_delegate:
            self.logger_delegate.fit(paths)

    def log_diagnostics(self, paths):
        if self.logger_delegate:
            self.logger_delegate.log_diagnostics(paths)

    def subsample_path(self, path):
        interval = self.subgoal_interval
        observations = path['observations']
        rewards = path['rewards']
        actions = path['actions']
        path_length = len(rewards)
        chunked_length = int(np.ceil(path_length * 1.0 / interval))
        padded_length = chunked_length * interval
        padded_rewards = np.append(rewards, np.zeros(padded_length - path_length))
        chunked_rewards = np.sum(
            np.reshape(padded_rewards, (chunked_length, interval)),
            axis=1
        )
        chunked_env_infos = tensor_utils.subsample_tensor_dict(path["env_infos"], interval)
        chunked_agent_infos = tensor_utils.subsample_tensor_dict(path["agent_infos"], interval)
        chunked_actions = actions[::interval]
        chunked_observations = observations[::interval]
        return dict(
            observations=chunked_observations,
            actions=chunked_actions,
            env_infos=chunked_env_infos,
            agent_infos=chunked_agent_infos,
            rewards=chunked_rewards,
        )

    def predict(self, path):
        path_length = len(path["rewards"])
        path = self.subsample_path(path)
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([path])
        N = flat_obs.shape[0]
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        high_probs = path["agent_infos"]["prob"]
        log_p_sprime_given_g_s = self.regressor.predict_log_likelihood(xs, ys)
        p_sprime_given_s = 0.
        n_subgoals = self.subgoal_space.n
        for goal in range(n_subgoals):
            goal_mat = np.tile(to_onehot(goal, n_subgoals), (N, 1))
            xs_goal = np.concatenate([flat_obs, goal_mat], axis=1)
            p_sprime_given_s += high_probs[:-1, goal] * np.exp(self.regressor.predict_log_likelihood(xs_goal, ys))
        bonuses = np.append(log_p_sprime_given_g_s - np.log(p_sprime_given_s + 1e-8), 0)
        bonuses = np.tile(
            np.expand_dims(bonuses, axis=1),
            (1, self.subgoal_interval)
        ).flatten()[:path_length]
        return bonuses

