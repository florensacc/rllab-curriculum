from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
from sandbox.rocky.tf.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.envs.base import EnvSpec
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.misc import tensor_utils
from rllab.misc import logger
from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
import numpy as np


class HiddenAwareParsimonyBonusEvaluator(BonusEvaluator, Serializable):
    def __init__(self, env_spec, policy, hidden_bonus_coeff=1., action_bonus_coeff=1., regressor_cls=None,
                 regressor_args=None):
        """
        :type env_spec: EnvSpec
        :type policy: StochasticGRUPolicy
        """
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy
        assert isinstance(env_spec.action_space, Discrete)
        if regressor_cls is None:
            regressor_cls = CategoricalMLPRegressor
        if regressor_args is None:
            regressor_args = dict()
        self.hidden_bonus_coeff = hidden_bonus_coeff
        self.action_bonus_coeff = action_bonus_coeff
        self.hidden_given_prev_regressor = regressor_cls(
            input_shape=(policy.n_subgoals,),
            output_dim=policy.n_subgoals,
            name="p_ht_given_ht1",
            **regressor_args
        )
        self.hidden_given_state_prev_regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + policy.n_subgoals,),
            output_dim=policy.n_subgoals,
            name="p_ht_given_st_ht1",
            **regressor_args
        )
        self.action_given_hidden_regressor = regressor_cls(
            input_shape=(policy.n_subgoals,),
            output_dim=env_spec.action_space.n,
            name="p_at_given_ht",
            **regressor_args
        )
        self.action_given_state_regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim,),
            output_dim=env_spec.action_space.n,
            name="p_at_given_st",
            **regressor_args
        )
        self.action_given_state_hidden_regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + policy.n_subgoals,),
            output_dim=env_spec.action_space.n,
            name="p_at_given_st_ht",
            **regressor_args
        )

    def fit(self, paths):
        obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
        logger.log("fitting p(ht|ht-1) regressor")
        self.hidden_given_prev_regressor.fit(prev_hiddens, hidden_states)
        logger.log("fitting p(ht|st,ht-1) regressor")
        self.hidden_given_state_prev_regressor.fit(np.concatenate([obs, prev_hiddens], axis=1), hidden_states)
        logger.log("fitting p(at|ht) regressor")
        self.action_given_hidden_regressor.fit(hidden_states, actions)
        logger.log("fitting p(at|st,ht) regressor")
        self.action_given_state_hidden_regressor.fit(np.concatenate([obs, hidden_states], axis=1), actions)
        logger.log("fitting p(at|st) regressor")
        self.action_given_state_regressor.fit(obs, actions)

    def predict(self, path):
        obs = path["observations"]
        actions = path["actions"]
        agent_infos = path["agent_infos"]
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
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
        return self.hidden_bonus_coeff * (log_p_ht_given_ht1 - log_p_ht_given_st_ht1) + \
               self.action_bonus_coeff * (log_p_at_given_ht - log_p_at_given_st_ht)

    def log_diagnostics(self, paths):
        obs = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        hidden_states = agent_infos["hidden_state"]
        prev_hiddens = agent_infos["prev_hidden"]
        ent_at_given_st = np.mean(-self.action_given_state_regressor.predict_log_likelihood(
            obs, actions))
        ent_at_given_ht = np.mean(-self.action_given_hidden_regressor.predict_log_likelihood(
            hidden_states, actions))
        ent_ht_given_ht1 = np.mean(-self.hidden_given_prev_regressor.predict_log_likelihood(
            prev_hiddens, hidden_states))
        ent_ht_given_st_ht1 = np.mean(-self.hidden_given_state_prev_regressor.predict_log_likelihood(
            np.concatenate([obs, prev_hiddens], axis=1), hidden_states))
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
