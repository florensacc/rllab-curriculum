from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import lasagne.layers as L
import theano.tensor as TT
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import GRUNetwork, MLP, GRULayer
from rllab.core.lasagne_layers import OpLayer
from rllab.distributions.recurrent_categorical import RecurrentCategorical
from rllab.misc import ext
from rllab.spaces.discrete import Discrete
from rllab.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


class TheanoCategoricalLookbackPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec):
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

        l_input = L.concat([l_obs, l_prev_action], name="input", axis=1)
        input_dim = obs_dim + action_dim

        l_hidden = L.DenseLayer(
            l_input,
            num_units=32,
            nonlinearity=TT.tanh,
        )

        l_prob = L.DenseLayer(
            l_hidden,
            num_units=action_dim,
            nonlinearity=TT.nnet.softmax,
        )

        self.l_input = l_input
        self.l_obs = l_obs
        self.l_prev_action = l_prev_action
        self.l_prob = l_prob

        self.input_dim = input_dim
        self.action_dim = action_dim

        self.prev_action = None
        self.dist = RecurrentCategorical(env_spec.action_space.n)

        self.f_prob = ext.compile_function(
            inputs=[self.l_obs.input_var, self.l_prev_action.input_var],
            outputs=L.get_output(self.l_prob),
            log_name="f_prob"
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
        prob = self.f_prob([flat_obs], [prev_action])[0]
        action = special.weighted_sample(prob, np.arange(self.action_dim))
        self.prev_action = self.action_space.flatten(action)
        return action, dict(prob=prob, prev_action=prev_action)

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_keys(self):
        return ["prev_action"]

    def reg_sym(self, obs_var, action_var, valid_var, state_info_vars, dist_info_vars, **kwargs):
        logli = self.distribution.log_likelihood_sym(action_var, dist_info_vars)
        mean_logli = TT.sum(logli * valid_var) / TT.sum(valid_var)
        action_saliency = TT.mean(TT.sum(TT.abs_(TT.grad(mean_logli, state_info_vars['prev_action'])), axis=-1))
        obs_saliency = TT.mean(TT.sum(TT.abs_(TT.grad(mean_logli, obs_var)), axis=-1))
        return obs_saliency - action_saliency
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
        if not hasattr(self, 'f_debug'):
            # return
            action_var = self.action_space.new_tensor_variable('action', extra_dims=2)
            obs_var = self.observation_space.new_tensor_variable('obs', extra_dims=2)
            # dist_info_vars_list = self.observation_space.new_tensor_variable('obs', extra_dims=2)
            valid_var = TT.matrix("valid")
            # dist_info_vars = {
            #     k: tf.placeholder(tf.float32, shape=[None, None] + list(shape), name=k)
            #     for k, shape in self.distribution.dist_info_specs
            #     }
            # dist_info_vars_list = [dist_info_vars[k] for k in self.distribution.dist_info_keys]

            state_info_vars = {
                k: ext.new_tensor(k, ndim=3, dtype='float32')
                for k in self.state_info_keys
                }
            state_info_vars_list = [state_info_vars[k] for k in self.state_info_keys]
            dist_info_vars = self.dist_info_sym(obs_var, dict(prev_action=state_info_vars["prev_action"]))

            logli = self.distribution.log_likelihood_sym(action_var, dist_info_vars)
            mean_logli = TT.sum(logli * valid_var) / TT.sum(valid_var)
            action_saliency = TT.mean(TT.sum(TT.abs_(TT.grad(mean_logli, state_info_vars['prev_action'])), axis=-1))
            obs_saliency = TT.mean(TT.sum(TT.abs_(TT.grad(mean_logli, obs_var)), axis=-1))

            self.f_debug = ext.compile_function(
                [obs_var, action_var, valid_var] + state_info_vars_list,
                [action_saliency, obs_saliency],
                log_name="f_debug"
            )

        agent_infos = samples_data['agent_infos']
        state_info_list = [agent_infos[k] for k in self.state_info_keys]

        action_saliency, obs_saliency = self.f_debug(
            samples_data['observations'], samples_data['actions'], samples_data['valids'], *state_info_list)
        from rllab.misc import logger
        logger.record_tabular('ActionSaliency', action_saliency)
        logger.record_tabular('ObsSaliency', obs_saliency)

        # W = self.l_flat_prob.W.eval()
        # W_hidden = W[:self.hidden_dim]
        # W_skipped = W[self.hidden_dim:]
        # logger.record_tabular('W_hidden.norm', np.linalg.norm(W_hidden))
        # logger.record_tabular('W_skipped.norm', np.linalg.norm(W_skipped))
