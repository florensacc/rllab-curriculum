


from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
from rllab.envs.base import EnvSpec
from rllab.core.serializable import Serializable
from sandbox.rocky.hrl.policies.two_part_policy.two_part_policy import TwoPartPolicy, DuelTwoPartPolicy
from sandbox.rocky.hrl.policies.two_part_policy.reflective_stochastic_mlp_policy import ReflectiveStochasticMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.hrl.policies.two_part_policy.categorical_mlp_policy import CategoricalMLPPolicy
import numpy as np
from rllab.misc import logger

from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.misc import tensor_utils
import theano.tensor as TT
import lasagne.layers as L
import pickle as pickle


class ExactHiddenRegressor(object):
    def __init__(self, env_spec, policy, exact_entropy, target_policy):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :type target_policy: StochasticGRUPolicy
        :type exact_entropy: bool
        """
        self.env_spec = env_spec
        self.exact_entropy = exact_entropy
        self.policy = policy
        self.target_policy = target_policy

    @classmethod
    def admissible(cls, policy):
        return (
            isinstance(policy, (TwoPartPolicy, DuelTwoPartPolicy)) and
            isinstance(policy.high_policy, ReflectiveStochasticMLPPolicy) and
            (
                isinstance(policy.high_policy.action_policy, CategoricalMLPPolicy) or
                isinstance(policy.high_policy.action_policy,
                           GaussianMLPPolicy) and policy.high_policy.gate_policy is None
            )
        )

    def fit(self, xs, ys):
        self.target_policy.set_param_values(self.policy.get_param_values())

    def predict_log_likelihood(self, xs, ys):
        high_policy = self.target_policy.high_policy
        split_idx = self.env_spec.observation_space.flat_dim
        obs, prev_hiddens = np.split(xs, [split_idx], axis=1)
        hidden_dist_info = high_policy.dist_info(obs, dict(prev_action=prev_hiddens))
        if self.exact_entropy:
            return -high_policy.distribution.entropy(hidden_dist_info)
        else:
            return high_policy.distribution.log_likelihood(ys, hidden_dist_info)

    def log_likelihood_sym(self, x_var, y_var):
        high_policy = self.target_policy.high_policy
        split_idx = self.env_spec.observation_space.flat_dim
        obs_var = x_var[:, :split_idx]
        prev_hidden_var = x_var[:, split_idx:]
        hidden_dist_info_sym = high_policy.dist_info_sym(obs_var, dict(prev_action=prev_hidden_var))
        if self.exact_entropy:
            return -high_policy.distribution.entropy_sym(hidden_dist_info_sym)
        else:
            return high_policy.distribution.log_likelihood_sym(y_var, hidden_dist_info_sym)


class ExactActionRegressor(object):
    def __init__(self, env_spec, policy, exact_entropy, target_policy):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :type target_policy: StochasticGRUPolicy
        :type exact_entropy: bool
        """
        self.env_spec = env_spec
        self.exact_entropy = exact_entropy
        self.policy = policy
        self.target_policy = target_policy

    def fit(self, xs, ys):
        self.target_policy.set_param_values(self.policy.get_param_values())

    @classmethod
    def admissible(cls, policy):
        return isinstance(policy, (TwoPartPolicy, DuelTwoPartPolicy))

    def predict_log_likelihood(self, xs, ys):
        low_policy = self.target_policy.low_policy
        action_dist_info = low_policy.dist_info(xs, dict())
        if self.exact_entropy:
            return -low_policy.distribution.entropy(action_dist_info)
        else:
            return low_policy.distribution.log_likelihood(ys, action_dist_info)

    def log_likelihood_sym(self, x_var, y_var):
        low_policy = self.target_policy.low_policy
        action_dist_info_sym = low_policy.dist_info_sym(x_var, dict())
        if self.exact_entropy:
            return -low_policy.distribution.entropy_sym(action_dist_info_sym)
        else:
            return low_policy.distribution.log_likelihood_sym(y_var, action_dist_info_sym)


class TwoPartBonusEvaluator(BonusEvaluator, Serializable):
    def __init__(
            self,
            env_spec,
            policy,
            bonus_coeff=1.,
            regressor_args=None,
            use_exact_regressor=True,
            exact_entropy=False,
            exact_stop_gradient=False,
    ):
        """
        :type env_spec: EnvSpec
        :type policy: TwoPartPolicy|DuelTwoPartPolicy
        :param use_exact_regressor: whether to use the exact quantity when available
        :param exact_stop_gradient: whether to take gradient through the policy when using it for the regressor
        """
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy

        assert isinstance(policy.high_policy, ReflectiveStochasticMLPPolicy)

        obs_dim = env_spec.observation_space.flat_dim
        hidden_obs_dim = obs_dim
        action_obs_dim = obs_dim
        action_dim = env_spec.action_space.flat_dim

        if regressor_args is None:
            regressor_args = dict()

        # Now, only measure I(ht;at|st) + I(ht;ht-1|st)

        if isinstance(policy.high_policy.action_space, Discrete):
            hidden_regressor_cls = CategoricalMLPRegressor
        elif isinstance(policy.high_policy.action_space, Box):
            hidden_regressor_cls = GaussianMLPRegressor
        else:
            raise NotImplementedError

        if isinstance(policy.low_policy.action_space, Discrete):
            action_regressor_cls = CategoricalMLPRegressor
        elif isinstance(policy.high_policy.action_space, Box):
            action_regressor_cls = GaussianMLPRegressor
        else:
            raise NotImplementedError

        if exact_stop_gradient:
            target_policy = pickle.loads(pickle.dumps(policy))
            target_policy.set_param_values(policy.get_param_values())
        else:
            target_policy = policy

        self.hidden_given_state_regressor = hidden_regressor_cls(
            input_shape=(hidden_obs_dim,),
            output_dim=policy.subgoal_dim,
            name="p_ht_given_st",
            **regressor_args
        )
        if use_exact_regressor and ExactHiddenRegressor.admissible(policy):
            self.hidden_given_state_prev_regressor = ExactHiddenRegressor(
                env_spec=env_spec,
                policy=policy,
                exact_entropy=exact_entropy,
                target_policy=target_policy,
            )
        else:
            self.hidden_given_state_prev_regressor = hidden_regressor_cls(
                input_shape=(hidden_obs_dim + policy.subgoal_dim,),
                output_dim=policy.subgoal_dim,
                name="p_ht_given_st_ht1",
                **regressor_args
            )

        self.action_given_state_regressor = action_regressor_cls(
            input_shape=(action_obs_dim,),
            output_dim=action_dim,
            name="p_at_given_st",
            **regressor_args
        )

        if use_exact_regressor and ExactActionRegressor.admissible(policy):
            self.action_given_state_hidden_regressor = ExactActionRegressor(
                env_spec=env_spec,
                policy=policy,
                exact_entropy=exact_entropy,
                target_policy=target_policy,
            )
        else:
            self.action_given_state_hidden_regressor = action_regressor_cls(
                input_shape=(action_obs_dim + policy.subgoal_dim,),
                output_dim=action_dim,
                name="p_at_given_st_ht",
                **regressor_args
            )

        self.bonus_coeff = bonus_coeff

    def fit(self, paths):
        raw_obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["high_action"]
        prev_hiddens = agent_infos["high_prev_action"]

        action_obs = raw_obs
        hidden_obs = raw_obs

        logger.log("fitting p(ht|st,ht-1) regressor")
        self.hidden_given_state_prev_regressor.fit(np.concatenate([hidden_obs, prev_hiddens], axis=1), hidden_states)
        logger.log("fitting p(ht|st) regressor")
        self.hidden_given_state_regressor.fit(hidden_obs, hidden_states)
        logger.log("fitting p(at|st,ht) regressor")
        self.action_given_state_hidden_regressor.fit(np.concatenate([action_obs, hidden_states], axis=1), actions)
        logger.log("fitting p(at|st) regressor")
        self.action_given_state_regressor.fit(action_obs, actions)

    def predict(self, path):
        raw_obs = path["observations"]
        actions = path["actions"]
        agent_infos = path["agent_infos"]
        hidden_states = agent_infos["high_action"]
        prev_hiddens = agent_infos["high_prev_action"]

        action_obs = raw_obs
        hidden_obs = raw_obs

        # The bonus will be computed as
        # log(p(at|ht,st)) - log(p(at|st)) + log(p(ht|ht-1,st)) - log(p(ht|st))
        log_p_at_given_st = self.action_given_state_regressor.predict_log_likelihood(
            action_obs, actions)
        log_p_at_given_st_ht = self.action_given_state_hidden_regressor.predict_log_likelihood(
            np.concatenate([action_obs, hidden_states], axis=1), actions)
        log_p_ht_given_st_ht1 = self.hidden_given_state_prev_regressor.predict_log_likelihood(
            np.concatenate([hidden_obs, prev_hiddens], axis=1), hidden_states)
        log_p_ht_given_st = self.hidden_given_state_regressor.predict_log_likelihood(
            hidden_obs, hidden_states)
        bonus = self.bonus_coeff * (log_p_at_given_st_ht - log_p_at_given_st) + \
                self.bonus_coeff * (log_p_ht_given_st_ht1 - log_p_ht_given_st)

        return bonus

    def bonus_sym(self, raw_obs_var, action_var, state_info_vars):
        assert raw_obs_var.ndim == 2

        hidden_state_var = state_info_vars["high_action"]
        prev_hidden_var = state_info_vars["high_prev_action"]

        action_obs_var = raw_obs_var
        hidden_obs_var = raw_obs_var

        # The bonus will be computed as
        # log(p(at|ht,st)) - log(p(at|st)) + log(p(ht|ht-1,st)) - log(p(ht|st))
        log_p_at_given_st = self.action_given_state_regressor.log_likelihood_sym(
            action_obs_var, action_var)
        log_p_at_given_st_ht = self.action_given_state_hidden_regressor.log_likelihood_sym(
            TT.concatenate([action_obs_var, hidden_state_var], axis=1), action_var)
        log_p_ht_given_st_ht1 = self.hidden_given_state_prev_regressor.log_likelihood_sym(
            TT.concatenate([hidden_obs_var, prev_hidden_var], axis=1), hidden_state_var)
        log_p_ht_given_st = self.hidden_given_state_regressor.log_likelihood_sym(
            hidden_obs_var, hidden_state_var)

        bonus = self.bonus_coeff * (log_p_at_given_st_ht - log_p_at_given_st) + \
                self.bonus_coeff * (log_p_ht_given_st_ht1 - log_p_ht_given_st)

        return bonus

    def log_diagnostics(self, paths):
        raw_obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["high_action"]
        prev_hiddens = agent_infos["high_prev_action"]

        action_obs = raw_obs
        hidden_obs = raw_obs

        ent_at_given_st = np.mean(-self.action_given_state_regressor.predict_log_likelihood(
            action_obs, actions))
        ent_ht_given_st_ht1 = np.mean(-self.hidden_given_state_prev_regressor.predict_log_likelihood(
            np.concatenate([hidden_obs, prev_hiddens], axis=1), hidden_states))
        ent_ht_given_st = np.mean(-self.hidden_given_state_regressor.predict_log_likelihood(
            hidden_obs, hidden_states))
        ent_at_given_st_ht = np.mean(-self.action_given_state_hidden_regressor.predict_log_likelihood(
            np.concatenate([action_obs, hidden_states], axis=1), actions))

        # so many terms lol
        logger.record_tabular("H(at|st)", ent_at_given_st)
        logger.record_tabular("H(at|st,ht)", ent_at_given_st_ht)
        logger.record_tabular("H(ht|st)", ent_ht_given_st)
        logger.record_tabular("H(ht|st,ht-1)", ent_ht_given_st_ht1)

        logger.record_tabular("I(at;ht|st)", ent_at_given_st - ent_at_given_st_ht)
        logger.record_tabular("I(ht;ht-1|st)", ent_ht_given_st - ent_ht_given_st_ht1)
