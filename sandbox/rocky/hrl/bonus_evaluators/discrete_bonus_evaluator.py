from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
from rllab.envs.base import EnvSpec
from rllab.core.serializable import Serializable
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
import numpy as np
from rllab.misc import logger

from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.spaces.discrete import Discrete
from rllab.misc import tensor_utils
import theano.tensor as TT
import lasagne.layers as L
import cPickle as pickle


class modes(object):
    MODE_MARGINAL_PARSIMONY = "min I(at;st)"
    MODE_HIDDEN_AWARE_PARSIMONY = "min I(at;st|ht) + I(ht;st|ht-1)"
    MODE_MI_FEUDAL = "max I(at;ht|st)"
    MODE_MI_FEUDAL_SYNC = "max I(at;ht|st) + I(ht;ht-1|st)"
    MODE_JOINT_MI_PARSIMONY = "max I(at;ht|st) + I(ht;ht-1|st) - I(at;st|ht) - I(ht;st|ht-1)"
    MODE_MI_FEUDAL_SYNC_NO_STATE = "max I(at;ht) + I(ht;ht-1)"
    MODE_BOTTLENECK_ONLY = "min I(st|st_raw,ht-1)"


MODES = modes()


class ExactHiddenRegressor(object):
    def __init__(self, env_spec, policy, exact_entropy, target_policy):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :type exact_entropy: bool
        """
        self.env_spec = env_spec
        self.exact_entropy = exact_entropy
        self.policy = policy
        self.target_policy = target_policy

    def fit(self, xs, ys):
        self.target_policy.set_param_values(self.policy.get_param_values())

    def predict_log_likelihood(self, xs, ys):
        # xs: concatenation of state and previous hidden state
        if self.target_policy.use_bottleneck:
            split_idx = self.target_policy.bottleneck_dim
        else:
            split_idx = self.env_spec.observation_space.flat_dim
        obs, prev_hiddens = np.split(xs, [split_idx], axis=1)
        hiddens = ys
        hidden_prob = self.target_policy.f_hidden_prob(obs, prev_hiddens)
        if self.exact_entropy:
            return -self.target_policy.hidden_dist.entropy(dict(prob=hidden_prob))
        else:
            return self.target_policy.hidden_dist.log_likelihood(hiddens, dict(prob=hidden_prob))

    def log_likelihood_sym(self, x_var, y_var):
        # xs: concatenation of state and previous hidden state
        if self.target_policy.use_bottleneck:
            split_idx = self.target_policy.bottleneck_dim
        else:
            split_idx = self.env_spec.observation_space.flat_dim
        obs_var = x_var[:, :split_idx]
        prev_hidden_var = x_var[:, split_idx:]
        hidden_var = TT.cast(y_var, 'int32')
        hidden_prob_var = L.get_output(
            self.target_policy.l_hidden_prob,
            {self.target_policy.l_obs: obs_var, self.target_policy.l_prev_hidden: prev_hidden_var}
        )
        if self.exact_entropy:
            return -self.target_policy.hidden_dist.entropy_sym(dict(prob=hidden_prob_var))
        else:
            return self.target_policy.hidden_dist.log_likelihood_sym(hidden_var, dict(prob=hidden_prob_var))


class ExactActionRegressor(object):
    def __init__(self, env_spec, policy, exact_entropy, target_policy):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :type exact_entropy: bool
        """
        self.env_spec = env_spec
        self.exact_entropy = exact_entropy
        self.policy = policy
        self.target_policy = target_policy

    def fit(self, xs, ys):
        self.target_policy.set_param_values(self.policy.get_param_values())

    def predict_log_likelihood(self, xs, ys):
        # xs: concatenation of state and hidden state
        if self.target_policy.use_bottleneck:
            split_idx = self.target_policy.bottleneck_dim
        else:
            split_idx = self.env_spec.observation_space.flat_dim
        obs, hiddens = np.split(xs, [split_idx], axis=1)
        actions = ys
        action_prob = self.target_policy.f_action_prob(obs, hiddens)
        if self.exact_entropy:
            return -self.target_policy.action_dist.entropy(dict(prob=action_prob))
        else:
            return self.target_policy.action_dist.log_likelihood(actions, dict(prob=action_prob))

    def log_likelihood_sym(self, x_var, y_var):
        # xs: concatenation of state and previous hidden state
        if self.target_policy.use_bottleneck:
            split_idx = self.target_policy.bottleneck_dim
        else:
            split_idx = self.env_spec.observation_space.flat_dim
        obs_var = x_var[:, :split_idx]
        hidden_var = x_var[:, split_idx:]
        action_var = TT.cast(y_var, 'int32')
        action_prob_var = L.get_output(
            self.target_policy.l_action_prob,
            {self.target_policy.l_obs: obs_var, self.target_policy.l_hidden: hidden_var}
        )
        if self.exact_entropy:
            return -self.target_policy.action_dist.entropy_sym(dict(prob=action_prob_var))
        else:
            return self.target_policy.action_dist.log_likelihood_sym(action_var, dict(prob=action_prob_var))


class ExactBottleneckRegressor(object):
    def __init__(self, env_spec, policy, exact_entropy, target_policy):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :type exact_entropy: bool
        """
        self.env_spec = env_spec
        self.exact_entropy = exact_entropy
        self.policy = policy
        self.target_policy = target_policy

    def fit(self, xs, ys):
        self.target_policy.set_param_values(self.policy.get_param_values())

    def predict_log_likelihood(self, xs, ys):
        # xs: concatenation of state and prev hidden state
        split_idx = self.env_spec.observation_space.flat_dim
        obs, prev_hiddens = np.split(xs, [split_idx], axis=1)
        bottlenecks = ys
        means, log_stds = self.target_policy.f_bottleneck_dist(obs, prev_hiddens)
        if self.exact_entropy:
            return -self.target_policy.bottleneck_dist.entropy(dict(mean=means, log_std=log_stds))
        else:
            return self.target_policy.bottleneck_dist.log_likelihood(bottlenecks, dict(mean=means, log_std=log_stds))

    def log_likelihood_sym(self, x_var, y_var):
        # xs: concatenation of state and previous hidden state
        split_idx = self.env_spec.observation_space.flat_dim
        obs_var = x_var[:, :split_idx]
        prev_hidden_var = x_var[:, split_idx:]
        bottleneck_var = y_var
        mean_var, log_std_var = L.get_output(
            [self.target_policy.l_bottleneck_mean, self.target_policy.l_bottleneck_log_std],
            {self.target_policy.l_raw_obs: obs_var, self.target_policy.l_prev_hidden: prev_hidden_var}
        )
        if self.exact_entropy:
            return -self.target_policy.bottleneck_dist.entropy_sym(dict(mean=mean_var, log_std=log_std_var))
        else:
            return self.target_policy.bottleneck_dist.log_likelihood_sym(bottleneck_var,
                                                                         dict(mean=mean_var, log_std=log_std_var))


def new_exact_regressor(env_spec, policy, regressor_type, exact_entropy, target_policy):
    if regressor_type == "p_ht_given_st_ht1":
        return ExactHiddenRegressor(env_spec, policy, exact_entropy, target_policy)
    elif regressor_type == "p_at_given_st_ht":
        return ExactActionRegressor(env_spec, policy, exact_entropy, target_policy)
    elif regressor_type == "p_st_given_st_raw_ht1":
        return ExactBottleneckRegressor(env_spec, policy, exact_entropy, target_policy)
    else:
        raise NotImplementedError


class DiscreteBonusEvaluator(BonusEvaluator, Serializable):
    def __init__(
            self,
            env_spec,
            policy,
            mode,
            bonus_coeff=1.,
            bottleneck_coeff=1.,
            hidden_regressor_cls=None,
            action_regressor_cls=None,
            bottleneck_regressor_cls=None,
            regressor_args=None,
            hidden_regressor_args=None,
            action_regressor_args=None,
            bottleneck_regressor_args=None,
            use_exact_regressor=False,
            exact_entropy=False,
            exact_stop_gradient=False,
    ):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        :param mode: Can be one of the following:
            - MODE_MARGINAL_PARSIMONY or min I(at;st)
                For this one, we assume that H(at) is constant, and compute the bonus as
                log(p(at)) - log(p(at|st))
            - MODE_HIDDEN_AWARE_PARSIMONY or min I(at;st|ht) + I(ht;st|ht-1)
                The bonus will be computed as:
                log(p(ht|ht-1)) - log(p(ht|ht-1,st)) + log(p(at|ht)) - log(p(at|ht,st))
            - MODE_MI_FEUDAL or max I(at;ht|st)
                The bonus will be computed as:
                log(p(at|ht,st)) - log(p(at|st))
            - MODE_MI_FEUDAL_SYNC or max I(at;ht|st) + I(ht;ht-1|st)
                The bonus will be computed as:
                log(p(at|ht,st)) - log(p(at|st)) + log(p(ht|ht-1,st)) - log(p(ht|st))
            - MODE_JOINT_MI_PARSIMONY or max I(at;ht|st) + I(ht;ht-1|st) - I(at;st|ht) - I(ht;st|ht-1)
                The bonus will be computed as:
                log(p(at|ht)) - log(p(at|st)) + log(p(ht|ht-1)) - log(p(ht|st))
            - MODE_MI_FEUDAL_SYNC_NO_STATE or max I(at;ht) + I(ht;ht-1)
                We ignore the marginal terms H(at) and H(ht), and compute the bonus as
                log(p(at|ht)) + log(p(ht|ht-1))
            - MODE_BOTTLENECK_ONLY or min I(st|st_raw,ht-1)
                This mode will only compute the bottleneck term as the bonus, if a bottleneck is used. Otherwise the
                bonus is 0
        :param use_exact_regressor: whether to use the exact quantity when available
        :param exact_stop_gradient: whether to take gradient through the policy when using it for the regressor
        :param exact_entropy: when exact quantity is available, whether to use log probability or entropy
        """
        # assert mode in MODES
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy
        assert isinstance(env_spec.action_space, Discrete)
        assert not policy.use_decision_nodes
        assert policy.use_bottleneck
        assert not policy.random_reset
        if regressor_args is None:
            regressor_args = dict()
        if hidden_regressor_cls is None:
            hidden_regressor_cls = CategoricalMLPRegressor
        if hidden_regressor_args is None:
            hidden_regressor_args = regressor_args
        if action_regressor_cls is None:
            action_regressor_cls = CategoricalMLPRegressor
        if action_regressor_args is None:
            action_regressor_args = regressor_args
        if bottleneck_regressor_cls is None:
            bottleneck_regressor_cls = GaussianMLPRegressor
        if bottleneck_regressor_args is None:
            bottleneck_regressor_args = regressor_args
        self.bonus_coeff = bonus_coeff
        self.bottleneck_coeff = bottleneck_coeff
        self.mode = mode
        if policy.use_bottleneck:
            obs_dim = policy.bottleneck_dim
        else:
            obs_dim = env_spec.observation_space.flat_dim
        self.hidden_given_prev_regressor = hidden_regressor_cls(
            input_shape=(policy.n_subgoals,),
            output_dim=policy.n_subgoals,
            name="p_ht_given_ht-1",
            **hidden_regressor_args
        )
        if exact_stop_gradient:
            target_policy = pickle.loads(pickle.dumps(policy))
            target_policy.set_param_values(policy.get_param_values())
        else:
            target_policy = policy
        if use_exact_regressor:
            self.hidden_given_state_prev_regressor = new_exact_regressor(
                env_spec, policy, "p_ht_given_st_ht1", exact_entropy, target_policy
            )
        else:
            self.hidden_given_state_prev_regressor = hidden_regressor_cls(
                input_shape=(obs_dim + policy.n_subgoals,),
                output_dim=policy.n_subgoals,
                name="p_ht_given_st_ht1",
                **hidden_regressor_args
            )
        self.hidden_given_state_regressor = hidden_regressor_cls(
            input_shape=(obs_dim,),
            output_dim=policy.n_subgoals,
            name="p_ht_given_st",
            **hidden_regressor_args
        )
        self.action_given_hidden_regressor = action_regressor_cls(
            input_shape=(policy.n_subgoals,),
            output_dim=env_spec.action_space.n,
            name="p_at_given_ht",
            **action_regressor_args
        )
        self.action_given_state_regressor = action_regressor_cls(
            input_shape=(obs_dim,),
            output_dim=env_spec.action_space.n,
            name="p_at_given_st",
            **action_regressor_args
        )
        if use_exact_regressor:
            self.action_given_state_hidden_regressor = new_exact_regressor(
                env_spec, policy, "p_at_given_st_ht", exact_entropy, target_policy
            )
        else:
            self.action_given_state_hidden_regressor = action_regressor_cls(
                input_shape=(obs_dim + policy.n_subgoals,),
                output_dim=env_spec.action_space.n,
                name="p_at_given_st_ht",
                **action_regressor_args
            )
        if policy.use_bottleneck:
            self.bottleneck_given_prev_regressor = bottleneck_regressor_cls(
                input_shape=(policy.n_subgoals,),
                output_dim=policy.bottleneck_dim,
                name="p_st_given_ht1",
                **bottleneck_regressor_args
            )
            if use_exact_regressor:
                self.bottleneck_given_state_prev_regressor = new_exact_regressor(
                    env_spec, policy, "p_st_given_st_raw_ht1", exact_entropy, target_policy
                )
            else:
                self.bottleneck_given_state_prev_regressor = bottleneck_regressor_cls(
                    input_shape=(env_spec.observation_space.flat_dim + policy.n_subgoals,),
                    output_dim=policy.bottleneck_dim,
                    name="p_st_given_st_raw_ht1",
                    **bottleneck_regressor_args
                )
        else:
            self.bottleneck_given_prev_regressor = None
            self.bottleneck_given_state_prev_regressor = None

    def fit(self, paths):
        raw_obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
        if self.policy.use_bottleneck:
            bottleneck = agent_infos["bottleneck"]
            logger.log("fitting p(st|ht1) regressor")
            self.bottleneck_given_prev_regressor.fit(prev_hiddens, bottleneck)
            logger.log("fitting p(st|st_raw_ht1) regressor")
            self.bottleneck_given_state_prev_regressor.fit(np.concatenate([raw_obs, prev_hiddens], axis=1), bottleneck)
            obs = bottleneck
        else:
            obs = raw_obs
        logger.log("fitting p(ht|ht-1) regressor")
        self.hidden_given_prev_regressor.fit(prev_hiddens, hidden_states)
        logger.log("fitting p(ht|st,ht-1) regressor")
        self.hidden_given_state_prev_regressor.fit(np.concatenate([obs, prev_hiddens], axis=1), hidden_states)
        logger.log("fitting p(ht|st) regressor")
        self.hidden_given_state_regressor.fit(obs, hidden_states)
        logger.log("fitting p(at|ht) regressor")
        self.action_given_hidden_regressor.fit(hidden_states, actions)
        logger.log("fitting p(at|st,ht) regressor")
        self.action_given_state_hidden_regressor.fit(np.concatenate([obs, hidden_states], axis=1), actions)
        logger.log("fitting p(at|st) regressor")
        self.action_given_state_regressor.fit(obs, actions)

    def predict(self, path):
        raw_obs = path["observations"]
        actions = path["actions"]
        agent_infos = path["agent_infos"]
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
        if self.policy.use_bottleneck:
            bottleneck = agent_infos["bottleneck"]
            obs = bottleneck
        else:
            bottleneck = None
            obs = raw_obs
        if self.mode == MODES.MODE_MARGINAL_PARSIMONY:
            # The bonus will be computed as - log(p(at|st))
            log_p_at_given_st = self.action_given_state_regressor.predict_log_likelihood(
                obs, actions)
            bonus = self.bonus_coeff * (-log_p_at_given_st)
        elif self.mode == MODES.MODE_HIDDEN_AWARE_PARSIMONY:
            # what we want is penalty = H(ht|ht-1) - H(ht|ht-1,st) + H(at|ht) - H(at|ht,st)
            # so the reward should be log(p(ht|ht-1)) - log(p(ht|ht-1,st)) + log(p(at|ht)) - log(p(at|ht,st))
            log_p_ht_given_ht1 = self.hidden_given_prev_regressor.predict_log_likelihood(
                prev_hiddens, hidden_states)
            log_p_ht_given_st_ht1 = self.hidden_given_state_prev_regressor.predict_log_likelihood(
                np.concatenate([obs, prev_hiddens], axis=1), hidden_states)
            log_p_at_given_ht = self.action_given_hidden_regressor.predict_log_likelihood(
                hidden_states, actions)
            log_p_at_given_st_ht = self.action_given_state_hidden_regressor.predict_log_likelihood(
                np.concatenate([obs, hidden_states], axis=1), actions)
            bonus = self.bonus_coeff * (log_p_ht_given_ht1 - log_p_ht_given_st_ht1) + \
                    self.bonus_coeff * (log_p_at_given_ht - log_p_at_given_st_ht)
        elif self.mode == MODES.MODE_MI_FEUDAL:
            # The bonus will be computed as log(p(at|ht,st)) - log(p(at|st))
            log_p_at_given_st = self.action_given_state_regressor.predict_log_likelihood(
                obs, actions)
            log_p_at_given_st_ht = self.action_given_state_hidden_regressor.predict_log_likelihood(
                np.concatenate([obs, hidden_states], axis=1), actions)
            bonus = self.bonus_coeff * (log_p_at_given_st_ht - log_p_at_given_st)
        elif self.mode == MODES.MODE_MI_FEUDAL_SYNC:
            # The bonus will be computed as
            # log(p(at|ht,st)) - log(p(at|st)) + log(p(ht|ht-1,st)) - log(p(ht|st))
            log_p_at_given_st = self.action_given_state_regressor.predict_log_likelihood(
                obs, actions)
            log_p_at_given_st_ht = self.action_given_state_hidden_regressor.predict_log_likelihood(
                np.concatenate([obs, hidden_states], axis=1), actions)
            log_p_ht_given_st_ht1 = self.hidden_given_state_prev_regressor.predict_log_likelihood(
                np.concatenate([obs, prev_hiddens], axis=1), hidden_states)
            log_p_ht_given_st = self.hidden_given_state_regressor.predict_log_likelihood(
                obs, hidden_states)
            bonus = self.bonus_coeff * (log_p_at_given_st_ht - log_p_at_given_st) + \
                    self.bonus_coeff * (log_p_ht_given_st_ht1 - log_p_ht_given_st)
        elif self.mode == MODES.MODE_JOINT_MI_PARSIMONY:
            # The bonus will be computed as
            # log(p(at|ht)) - log(p(at|st)) + log(p(ht|ht-1)) - log(p(ht|st))
            log_p_at_given_st = self.action_given_state_regressor.predict_log_likelihood(
                obs, actions)
            log_p_at_given_ht = self.action_given_hidden_regressor.predict_log_likelihood(
                hidden_states, actions)
            log_p_ht_given_ht1 = self.hidden_given_prev_regressor.predict_log_likelihood(
                prev_hiddens, hidden_states)
            log_p_ht_given_st = self.hidden_given_state_regressor.predict_log_likelihood(
                obs, hidden_states)
            bonus = self.bonus_coeff * (log_p_at_given_ht - log_p_at_given_st) + \
                    self.bonus_coeff * (log_p_ht_given_ht1 - log_p_ht_given_st)
        elif self.mode == MODES.MODE_MI_FEUDAL_SYNC_NO_STATE:
            # The bonus will be computed as
            # log(p(at|ht)) + log(p(ht|ht-1))
            log_p_at_given_ht = self.action_given_hidden_regressor.predict_log_likelihood(
                hidden_states, actions)
            log_p_ht_given_ht1 = self.hidden_given_prev_regressor.predict_log_likelihood(
                prev_hiddens, hidden_states)
            bonus = self.bonus_coeff * (log_p_at_given_ht + log_p_ht_given_ht1)
        elif self.mode == MODES.MODE_BOTTLENECK_ONLY:
            bonus = 0.
        else:
            raise NotImplementedError

        if self.policy.use_bottleneck:
            # need to add the information bottleneck term
            # min I(st;st_raw|ht-1) = H(st|ht-1) - H(st|st_raw,ht-1)
            # hence the bonus should be
            # log(p(st|ht-1)) - log(p(st|st_raw,ht-1))
            log_p_st_given_ht1 = self.bottleneck_given_prev_regressor.predict_log_likelihood(
                prev_hiddens, bottleneck)
            log_p_st_given_st_raw_ht1 = self.bottleneck_given_state_prev_regressor.predict_log_likelihood(
                np.concatenate([raw_obs, prev_hiddens], axis=1), bottleneck
            )
            bonus += self.bottleneck_coeff * (log_p_st_given_ht1 - log_p_st_given_st_raw_ht1)
        return bonus

    def bonus_sym(self, raw_obs_var, action_var, state_info_vars):
        hidden_state_var = state_info_vars["hidden_state"]
        prev_hidden_var = state_info_vars["prev_hidden"]
        if self.policy.use_bottleneck:
            dist_info = self.policy.dist_info_sym(raw_obs_var, state_info_vars)
            bottleneck_mean = dist_info["bottleneck_mean"]
            bottleneck_log_std = dist_info["bottleneck_log_std"]
            bottleneck_epsilon = state_info_vars["bottleneck_epsilon"]
            bottleneck_var = bottleneck_epsilon * TT.exp(bottleneck_log_std) + bottleneck_mean
            obs_var = bottleneck_var
        else:
            bottleneck_var = None
            obs_var = raw_obs_var
        if self.mode == MODES.MODE_MARGINAL_PARSIMONY:
            # The bonus will be computed as - log(p(at|st))
            log_p_at_given_st = self.action_given_state_regressor.log_likelihood_sym(obs_var, action_var)
            bonus = self.bonus_coeff * (-log_p_at_given_st)
        elif self.mode == MODES.MODE_HIDDEN_AWARE_PARSIMONY:
            # what we want is penalty = H(ht|ht-1) - H(ht|ht-1,st) + H(at|ht) - H(at|ht,st)
            # so the reward should be log(p(ht|ht-1)) - log(p(ht|ht-1,st)) + log(p(at|ht)) - log(p(at|ht,st))
            log_p_ht_given_ht1 = self.hidden_given_prev_regressor.log_likelihood_sym(
                prev_hidden_var, hidden_state_var)
            log_p_ht_given_st_ht1 = self.hidden_given_state_prev_regressor.log_likelihood_sym(
                TT.concatenate([obs_var, hidden_state_var], axis=1), hidden_state_var)
            log_p_at_given_ht = self.action_given_hidden_regressor.log_likelihood_sym(
                hidden_state_var, action_var)
            log_p_at_given_st_ht = self.action_given_state_hidden_regressor.log_likelihood_sym(
                TT.concatenate([obs_var, hidden_state_var], axis=1), action_var)
            bonus = self.bonus_coeff * (log_p_ht_given_ht1 - log_p_ht_given_st_ht1) + \
                    self.bonus_coeff * (log_p_at_given_ht - log_p_at_given_st_ht)
        elif self.mode == MODES.MODE_MI_FEUDAL:
            # The bonus will be computed as log(p(at|ht,st)) - log(p(at|st))
            log_p_at_given_st = self.action_given_state_regressor.log_likelihood_sym(
                obs_var, action_var)
            log_p_at_given_st_ht = self.action_given_state_hidden_regressor.log_likelihood_sym(
                TT.concatenate([obs_var, hidden_state_var], axis=1), action_var)
            bonus = self.bonus_coeff * (log_p_at_given_st_ht - log_p_at_given_st)
        elif self.mode == MODES.MODE_MI_FEUDAL_SYNC:
            # The bonus will be computed as
            # log(p(at|ht,st)) - log(p(at|st)) + log(p(ht|ht-1,st)) - log(p(ht|st))
            log_p_at_given_st = self.action_given_state_regressor.log_likelihood_sym(
                obs_var, action_var)
            log_p_at_given_st_ht = self.action_given_state_hidden_regressor.log_likelihood_sym(
                TT.concatenate([obs_var, hidden_state_var], axis=1), action_var)
            log_p_ht_given_st_ht1 = self.hidden_given_state_prev_regressor.log_likelihood_sym(
                TT.concatenate([obs_var, prev_hidden_var], axis=1), hidden_state_var)
            log_p_ht_given_st = self.hidden_given_state_regressor.log_likelihood_sym(
                obs_var, hidden_state_var)
            bonus = self.bonus_coeff * (log_p_at_given_st_ht - log_p_at_given_st) + \
                    self.bonus_coeff * (log_p_ht_given_st_ht1 - log_p_ht_given_st)
        elif self.mode == MODES.MODE_JOINT_MI_PARSIMONY:
            # The bonus will be computed as
            # log(p(at|ht)) - log(p(at|st)) + log(p(ht|ht-1)) - log(p(ht|st))
            log_p_at_given_st = self.action_given_state_regressor.log_likelihood_sym(
                obs_var, action_var)
            log_p_at_given_ht = self.action_given_hidden_regressor.log_likelihood_sym(
                hidden_state_var, action_var)
            log_p_ht_given_ht1 = self.hidden_given_prev_regressor.log_likelihood_sym(
                prev_hidden_var, hidden_state_var)
            log_p_ht_given_st = self.hidden_given_state_regressor.log_likelihood_sym(
                obs_var, hidden_state_var)
            bonus = self.bonus_coeff * (log_p_at_given_ht - log_p_at_given_st) + \
                    self.bonus_coeff * (log_p_ht_given_ht1 - log_p_ht_given_st)
        elif self.mode == MODES.MODE_MI_FEUDAL_SYNC_NO_STATE:
            # The bonus will be computed as
            # log(p(at|ht)) + log(p(ht|ht-1))
            log_p_at_given_ht = self.action_given_hidden_regressor.log_likelihood_sym(
                hidden_state_var, action_var)
            log_p_ht_given_ht1 = self.hidden_given_prev_regressor.log_likelihood_sym(
                prev_hidden_var, hidden_state_var)
            bonus = self.bonus_coeff * (log_p_at_given_ht + log_p_ht_given_ht1)
        elif self.mode == MODES.MODE_BOTTLENECK_ONLY:
            bonus = 0.
        else:
            raise NotImplementedError

        if self.policy.use_bottleneck:
            # need to add the information bottleneck term
            # min I(st;st_raw|ht-1) = H(st|ht-1) - H(st|st_raw,ht-1)
            # hence the bonus should be
            # log(p(st|ht-1)) - log(p(st|st_raw,ht-1))
            log_p_st_given_ht1 = self.bottleneck_given_prev_regressor.log_likelihood_sym(
                prev_hidden_var, bottleneck_var)
            log_p_st_given_st_raw_ht1 = self.bottleneck_given_state_prev_regressor.log_likelihood_sym(
                TT.concatenate([raw_obs_var, prev_hidden_var], axis=1), bottleneck_var
            )
            bonus += self.bottleneck_coeff * (log_p_st_given_ht1 - log_p_st_given_st_raw_ht1)
        return bonus

    def log_diagnostics(self, paths):
        raw_obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
        if self.policy.use_bottleneck:
            bottleneck = agent_infos["bottleneck"]
            obs = bottleneck
        else:
            bottleneck = None
            obs = raw_obs
        ent_at_given_st = np.mean(-self.action_given_state_regressor.predict_log_likelihood(
            obs, actions))
        ent_at_given_ht = np.mean(-self.action_given_hidden_regressor.predict_log_likelihood(
            hidden_states, actions))
        ent_ht_given_ht1 = np.mean(-self.hidden_given_prev_regressor.predict_log_likelihood(
            prev_hiddens, hidden_states))
        ent_ht_given_st_ht1 = np.mean(-self.hidden_given_state_prev_regressor.predict_log_likelihood(
            np.concatenate([obs, prev_hiddens], axis=1), hidden_states))
        ent_ht_given_st = np.mean(-self.hidden_given_state_regressor.predict_log_likelihood(
            obs, hidden_states))
        ent_at_given_st_ht = np.mean(-self.action_given_state_hidden_regressor.predict_log_likelihood(
            np.concatenate([obs, hidden_states], axis=1), actions))
        # so many terms lol
        logger.record_tabular("H(at|st)", ent_at_given_st)
        logger.record_tabular("H(at|ht)", ent_at_given_ht)
        logger.record_tabular("H(at|st,ht)", ent_at_given_st_ht)
        logger.record_tabular("H(ht|ht-1)", ent_ht_given_ht1)
        logger.record_tabular("H(ht|st,ht-1)", ent_ht_given_st_ht1)
        logger.record_tabular("I(at;st|ht)", ent_at_given_ht - ent_at_given_st_ht)
        logger.record_tabular("I(ht;st|ht-1)", ent_ht_given_ht1 - ent_ht_given_st_ht1)
        logger.record_tabular("I(at;ht|st)", ent_at_given_st - ent_at_given_st_ht)
        logger.record_tabular("I(ht;ht-1|st)", ent_ht_given_st - ent_ht_given_st_ht1)
        if self.policy.use_bottleneck:
            ent_st_given_ht1 = np.mean(-self.bottleneck_given_prev_regressor.predict_log_likelihood(
                prev_hiddens, bottleneck
            ))
            ent_st_given_st_raw_ht1 = np.mean(-self.bottleneck_given_state_prev_regressor.predict_log_likelihood(
                np.concatenate([raw_obs, prev_hiddens], axis=1), bottleneck
            ))
            logger.record_tabular("H(st|ht-1)", ent_st_given_ht1)
            logger.record_tabular("H(st|st_raw,ht-1)", ent_st_given_st_raw_ht1)
            logger.record_tabular("I(st;st_raw|ht-1)", ent_st_given_ht1 - ent_st_given_st_raw_ht1)
