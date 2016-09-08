


import numpy as np
import lasagne.layers as L
import theano.tensor as TT
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork, MLP, GRULayer
from rllab.core.lasagne_layers import OpLayer
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.misc import ext
from rllab.misc import tensor_utils
from rllab.misc import logger
from rllab.spaces.discrete import Discrete
from rllab.policies.base import StochasticPolicy
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


class TheanoCategoricalLookbackPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            bottleneck_dim=32,
            mi_coeff=0.0,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(TheanoCategoricalLookbackPolicy, self).__init__(env_spec)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        l_obs = L.InputLayer(shape=(None, obs_dim), name="obs")
        l_prev_action = L.InputLayer(shape=(None, action_dim), name="action")

        l_bottleneck = L.DenseLayer(
            l_obs,
            num_units=bottleneck_dim,
            nonlinearity=TT.tanh,
        )
        l_joint = L.concat([l_bottleneck, l_prev_action], axis=1)

        l_hidden = L.DenseLayer(
            l_joint,
            num_units=32,
            nonlinearity=TT.tanh,
        )

        l_prob = L.DenseLayer(
            l_hidden,
            num_units=action_dim,
            nonlinearity=TT.nnet.softmax,
        )

        self.bottleneck_dim = bottleneck_dim
        self.l_obs = l_obs
        self.l_prev_action = l_prev_action
        self.l_bottleneck = l_bottleneck
        self.l_prob = l_prob

        self.action_dim = action_dim
        self.mi_coeff = mi_coeff

        self.prev_action = None
        self.dist = RecurrentCategorical(env_spec.action_space.n)

        self.f_prob_bottleneck = ext.compile_function(
            inputs=[self.l_obs.input_var, self.l_prev_action.input_var],
            outputs=L.get_output([self.l_prob, self.l_bottleneck]),
            log_name="f_prob_bottleneck"
        )

        LasagnePowered.__init__(self, [l_prob])

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        flat_obs_var = obs_var.reshape((-1, obs_var.shape[-1]))
        prev_action_var = state_info_vars["prev_action"]
        flat_prev_action_var = prev_action_var.reshape((-1, prev_action_var.shape[-1]))
        flat_prob = L.get_output(self.l_prob, {self.l_obs: flat_obs_var, self.l_prev_action: flat_prev_action_var})
        prob = flat_prob.reshape((obs_var.shape[0], obs_var.shape[1], -1))
        return dict(prob=prob)

    def reset(self):
        self.prev_action = np.zeros((self.action_dim,))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prev_action = self.prev_action
        prob, bottleneck = [x[0] for x in self.f_prob_bottleneck([flat_obs], [prev_action])]
        action = special.weighted_sample(prob, np.arange(self.action_dim))
        self.prev_action = self.action_space.flatten(action)
        return action, dict(prob=prob, prev_action=prev_action, bottleneck=bottleneck)

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_keys(self):
        return ["prev_action", "reward_bonus"]

    def reg_sym(self, state_info_vars, action_var, dist_info_vars, old_dist_info_vars, valid_var, **kwargs):
        lr = self.distribution.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        reward_bonus = state_info_vars["reward_bonus"]
        reward_bonus = reward_bonus.reshape([reward_bonus.shape[0], reward_bonus.shape[1]])
        return self.mi_coeff * -TT.sum(lr * reward_bonus * valid_var) / TT.sum(valid_var)

    # def reg_sym(self, obs_var, action_var, valid_var, state_info_vars, dist_info_vars, **kwargs):
    #     logli = self.distribution.log_likelihood_sym(action_var, dist_info_vars)
    #     mean_logli = TT.sum(logli * valid_var) / TT.sum(valid_var)
    #     action_saliency = TT.mean(TT.sum(TT.abs_(TT.grad(mean_logli, state_info_vars['prev_action'])), axis=-1))
    #     obs_saliency = TT.mean(TT.sum(TT.abs_(TT.grad(mean_logli, obs_var)), axis=-1))
    #     return obs_saliency - action_saliency

    #     dist_info_vars = self.dist_info_sym(obs_var, dict(prev_action=state_info_vars["prev_action_probs"]))
    #     logli = self.distribution.log_likelihood_sym(action_var, dist_info_vars)
    #     mean_logli = TT.sum(logli * valid_var) / TT.sum(valid_var)
    #     saliency = TT.mean(TT.square(TT.grad(mean_logli, state_info_vars['prev_action_probs'])))
    #     return - saliency
    #     # import ipdb; ipdb.set_trace()
    #     # self.dist_info_sym(obs_var, state_info_vars)
    #
    #     # regularize the change in the hidden units
    #     # return 0
    #     # pass


    def log_full_diagnostics(self, samples_data):
        # fit the regressor for p(at|zt)
        paths = samples_data["paths"]
        agent_infos = tensor_utils.concat_tensor_dict_list([p["agent_infos"] for p in paths])
        actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
        bottlenecks = agent_infos["bottleneck"]
        probs = agent_infos["prob"]

        if not hasattr(self, 'regressor'):
            self.regressor = CategoricalMLPRegressor(
                input_shape=(self.bottleneck_dim,),
                output_dim=self.action_dim,
                hidden_nonlinearity=TT.tanh,
                use_trust_region=False,
                name="p(at|zt)"
            )
        logger.log("fitting p(at|zt) regressor...")

        # only fit the valid part
        self.regressor.fit(bottlenecks, actions)

        all_bottlenecks = samples_data["agent_infos"]["bottleneck"]
        all_actions = samples_data["actions"]
        flat_bottlenecks = all_bottlenecks.reshape((-1, self.bottleneck_dim))
        flat_actions = all_actions.reshape((-1, self.action_dim))
        log_p_at_given_zt = self.regressor.predict_log_likelihood(flat_bottlenecks, flat_actions)
        log_p_at_given_zt = log_p_at_given_zt.reshape((all_bottlenecks.shape[0], all_bottlenecks.shape[1]))
        log_p_at_given_zt_aprev = np.log(np.sum(all_actions * samples_data["agent_infos"]["prob"], axis=-1) + 1e-8)

        mi_bonus = log_p_at_given_zt_aprev - log_p_at_given_zt

        samples_data["agent_infos"]["reward_bonus"] = np.expand_dims(mi_bonus, -1)

        ent_at_given_zt = np.mean(-self.regressor.predict_log_likelihood(bottlenecks, actions))
        ent_at_given_zt_aprev = np.mean(np.sum(- probs * np.log(probs + 1e-8), axis=-1))

        mi = ent_at_given_zt - ent_at_given_zt_aprev
        logger.record_tabular("H(at|zt)", ent_at_given_zt)
        logger.record_tabular("H(at|zt,at-1)", ent_at_given_zt_aprev)
        logger.record_tabular("I(at;at-1|zt)", mi)
