from __future__ import print_function
from __future__ import absolute_import

from rllab import config

from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
from rllab.envs.base import EnvSpec
from rllab.core.serializable import Serializable
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
import numpy as np
from rllab.misc import logger

from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.hrl.density_estimators.gaussian_density_estimator import GaussianDenstiyEstimator
from rllab.spaces.discrete import Discrete
from rllab.misc import tensor_utils


class modes(object):
    MODE_MARGINAL_PARSIMONY = "min I(at;st)"
    MODE_HIDDEN_AWARE_PARSIMONY = "min I(at;st|ht) + I(ht;st|ht-1)"
    MODE_MI_FEUDAL = "max I(at;ht|st)"
    MODE_MI_FEUDAL_SYNC = "max I(at;ht|st) + I(ht;ht-1|st)"
    MODE_JOINT_MI_PARSIMONY = "max I(at;ht|st) + I(ht;ht-1|st) - I(at;st|ht) - I(ht;st|ht-1)"
    MODE_MI_FEUDAL_SYNC_NO_STATE = "max I(at;ht) + I(ht;ht-1)"


MODES = modes()


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
            bottleneck_density_estimator_cls=None,
            regressor_args=None,
            hidden_regressor_args=None,
            action_regressor_args=None,
            bottleneck_regressor_args=None,
            bottleneck_density_estimator_args=None,
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
        """
        # assert mode in MODES
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy
        assert isinstance(env_spec.action_space, Discrete)
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
        if bottleneck_density_estimator_cls is None:
            bottleneck_density_estimator_cls = GaussianDenstiyEstimator
        if bottleneck_regressor_args is None:
            bottleneck_regressor_args = regressor_args
        if bottleneck_density_estimator_args is None:
            bottleneck_density_estimator_args = dict()
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
            name="p_ht_given_ht1",
            **hidden_regressor_args
        )
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
        self.action_given_state_hidden_regressor = action_regressor_cls(
            input_shape=(obs_dim + policy.n_subgoals,),
            output_dim=env_spec.action_space.n,
            name="p_at_given_st_ht",
            **action_regressor_args
        )
        if policy.use_bottleneck:
            self.bottleneck_density_estimator = bottleneck_density_estimator_cls(
                data_dim=policy.bottleneck_dim,
                name="p_bt",
                **bottleneck_density_estimator_args
            )
            self.bottleneck_given_state_regressor = bottleneck_regressor_cls(
                input_shape=(env_spec.observation_space.flat_dim,),
                output_dim=policy.bottleneck_dim,
                name="p_bt_given_st",
                **bottleneck_regressor_args
            )
        else:
            self.bottleneck_density_estimator = None
            self.bottleneck_given_state_regressor = None

    def fit(self, paths):
        raw_obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
        if self.policy.use_bottleneck:
            bottleneck = agent_infos["bottleneck"]
            logger.log("fitting p(st) density estimator")
            self.bottleneck_density_estimator.fit(bottleneck)
            logger.log("fitting p(st|st_raw) regressor")
            self.bottleneck_given_state_regressor.fit(raw_obs, bottleneck)
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
        else:
            raise NotImplementedError

        if self.policy.use_bottleneck:
            # need to add the information bottleneck term
            # min I(st;st_raw) = H(st) - H(st|st_raw)
            # hence the bonus should be
            # log(p(st)) - log(p(st|st_raw))
            log_p_st = self.bottleneck_density_estimator.predict_log_likelihood(bottleneck)
            log_p_st_given_st_raw = self.bottleneck_given_state_regressor.predict_log_likelihood(
                raw_obs, bottleneck
            )
            bonus += self.bottleneck_coeff * (log_p_st - log_p_st_given_st_raw)
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
        logger.record_tabular("approx_H(at|st)", ent_at_given_st)
        logger.record_tabular("approx_H(at|ht)", ent_at_given_ht)
        logger.record_tabular("approx_H(at|st,ht)", ent_at_given_st_ht)
        logger.record_tabular("approx_H(ht|ht-1)", ent_ht_given_ht1)
        logger.record_tabular("approx_H(ht|st,ht-1)", ent_ht_given_st_ht1)
        logger.record_tabular("approx_I(at;st|ht)", ent_at_given_ht - ent_at_given_st_ht)
        logger.record_tabular("approx_I(ht;st|ht-1)", ent_ht_given_ht1 - ent_ht_given_st_ht1)
        logger.record_tabular("approx_I(at;ht|st)", ent_at_given_st - ent_at_given_st_ht)
        logger.record_tabular("approx_I(ht;ht-1|st)", ent_ht_given_st - ent_ht_given_st_ht1)
        if self.policy.use_bottleneck:
            ent_st = np.mean(-self.bottleneck_density_estimator.predict_log_likelihood(bottleneck))
            ent_st_given_st_raw = np.mean(-self.bottleneck_given_state_regressor.predict_log_likelihood(
                raw_obs, bottleneck
            ))
            logger.record_tabular("approx_H(st)", ent_st)
            logger.record_tabular("approx_H(st|st_raw)", ent_st_given_st_raw)
            logger.record_tabular("approx_I(st;st_raw)", ent_st - ent_st_given_st_raw)
