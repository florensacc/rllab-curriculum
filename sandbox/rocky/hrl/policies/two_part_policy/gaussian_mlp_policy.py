from __future__ import print_function
from __future__ import absolute_import

import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano
import theano.tensor as TT


floatX = np.cast[theano.config.floatX]


class GaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            min_std=1e-6,
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
            mean_network=None,
            std_network=None,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :return:
        """
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        if mean_network is None:
            mean_network = MLP(
                input_shape=(obs_dim,),
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name="mean_network"
            )

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_layer.input_var

        if std_network is not None:
            l_log_std = std_network.output_layer
        else:
            if adaptive_std:
                if std_share_network:
                    # if share network, only add a linear layer parallel to the last layer of action mean network
                    l_log_std = L.DenseLayer(
                        mean_network.layers[-2],
                        num_units=action_dim,
                        nonlinearity=None,
                        W=LI.GlorotUniform(gain=0.1),
                    )
                else:
                    std_network = MLP(
                        input_shape=(obs_dim,),
                        input_layer=mean_network.input_layer,
                        output_dim=action_dim,
                        hidden_sizes=std_hidden_sizes,
                        hidden_nonlinearity=std_hidden_nonlinearity,
                        output_nonlinearity=None,
                        name="std_network",
                        output_W_init=LI.GlorotUniform(gain=0.1),
                    )
                    l_log_std = std_network.output_layer
            else:
                l_log_std = ParamLayer(
                    mean_network.input_layer,
                    num_units=action_dim,
                    param=lasagne.init.Constant(np.log(init_std)),
                    name="output_log_std",
                    trainable=learn_std,
                )

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if min_std is not None:
            log_std_var = TT.maximum(log_std_var, floatX(np.log(min_std)))

        self.mean_network = mean_network
        self.min_std = min_std
        self.mean_var, self.log_std_var = mean_var, log_std_var
        self.l_mean = l_mean
        self.l_log_std = l_log_std
        self.dist = DiagonalGaussian(action_dim)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMLPPolicy, self).__init__(env_spec)

        self.f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var, log_std_var = L.get_output([self.l_mean, self.l_log_std], obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, floatX(np.log(self.min_std)))
        return dict(mean=mean_var, log_std=log_std_var)

    @property
    def state_info_keys(self):
        """
        This only gets used when we want to reparametrize the action distribution
        """
        return ["epsilon"]

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self.f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std, epsilon=rnd)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self.f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds, epsilon=rnd)

    def get_reparam_action_sym(self, obs_var, state_info_vars):
        """
        Given observations and the state of the policy, compute the symbolic reparametrized actions
        """
        dist_info_vars = self.dist_info_sym(obs_var, state_info_vars)
        mean = dist_info_vars["mean"]
        log_std = dist_info_vars["log_std"]
        rnd = state_info_vars["epsilon"]
        return mean + TT.exp(log_std) * rnd

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self.dist
