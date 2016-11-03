import joblib
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from sandbox.dave.rllab.core.alexnet import AlexNet
from sandbox.dave.rllab.spaces import Box
from sandbox.dave.rllab.core.lasagne_layers import *
from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT

import pdb

class DepthGaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
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
            output_gain=1.0,
            num_relu=0,
            pkl_path=None,
            json_path=None,
            npz_path=None,
            seed=0,
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

        self.pkl_path = pkl_path
        self.json_path = json_path
        self.npz_path = npz_path

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        l_input = L.InputLayer(shape=(None,) + (obs_dim,), name="input_layer")

        # create network
        if mean_network is None:
            l_obs_robot = CropLayer(l_input, start_index=None, end_index=23)
            l_image = CropLayer(l_input, start_index=23, end_index=None)
            l_image = L.ReshapeLayer(l_image, (-1, 3, 227, 227))
            conv_network = AlexNet(input_layer=l_image)
            # l_input = L.ConcatLayer([l_obs_robot, conv_network.input_layer])

            l_conv_out = conv_network.output_layer
            l_input_mean = L.ConcatLayer([l_obs_robot, l_conv_out])
            mean_network = MLP(
                input_layer=l_input_mean,
                output_dim=action_dim,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
            )

        layers_mean = mean_network.layers
        l_mean = mean_network.output_layer
        obs_var = l_input.input_var


        if std_network is not None:
            l_log_std = std_network.output_layer
        else:
            if adaptive_std:
                std_network = MLP(
                    input_shape=(obs_dim,),
                    input_layer=l_input,
                    output_dim=action_dim,
                    hidden_sizes=std_hidden_sizes,
                    hidden_nonlinearity=std_hidden_nonlinearity,
                    output_nonlinearity=None,
                )
                l_log_std = std_network.output_layer
                layers_log_std = std_network.layers
            else:
                l_log_std = ParamLayer(
                    incoming=l_input,
                    num_units=action_dim,
                    param=lasagne.init.Constant(np.log(init_std)),
                    name="output_log_std",
                    trainable=learn_std,
                )
                layers_log_std = [l_log_std]

        self.min_std = min_std

        self._layers_old = conv_network.layers  #this returns a list with the old layers

        if self.pkl_path is not None:
            data = joblib.load(self.pkl_path)
            warm_params = data['policy'].get_params_internal()
            self.set_params_old(warm_params)

        elif self.npz_path is not None:
            warm_params = dict(np.load(self.npz_path))
            self.set_params_old(warm_params)

        # for i, layer in enumerate(mean_network.layers):
        #     if i > 0:
        #         mean_network._layers[i].params[mean_network._layers[i].W].remove("trainable")
        #         mean_network._layers[i].params[mean_network._layers[i].b].remove("trainable")
        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(min_std))

        self._mean_var, self._log_std_var, self._obs_var = mean_var, log_std_var, obs_var
        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = DiagonalGaussian(action_dim)

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(DepthGaussianMLPPolicy, self).__init__(env_spec)
        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

    @overrides
    def get_params_internal(self, **tags):
        return L.get_all_params(
            L.concat(self._output_layers),
            trainable=True,
            #**tags
        )#, key=lambda x: x.name)

    def get_params_old(self):
        params = []
        names = []
        for name, value in self._layers_old.items():
            params += value.get_params()
            names.append(name)
        return params, names

    # another way will be to do as in parametrized.py and flatten_tensors (in numpy). But with this I check names
    def set_params_old(self, snn_params):
        local_params, names = self.get_params_old()
        if type(snn_params) is dict:
            params_value_by_name = snn_params
        else:
            raise TypeError

        for param in local_params:
            param.set_value(params_value_by_name[param.name])

    def dist_info_sym(self, obs_var, state_info_vars=None):
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], obs_var)
        if self.min_std is not None:
            log_std_var = TT.maximum(log_std_var, np.log(self.min_std))
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        means, log_stds = self._f_dist(flat_obs)
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars["mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars["mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (TT.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * TT.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))
        logger.record_tabular('Weight1', L.get_all_param_values(self.train_out)[1][1])
        logger.record_tabular('Weight2', L.get_all_param_values(self.train_out)[1][2])
        logger.record_tabular('Weight3', L.get_all_param_values(self.train_out)[1][3])
        logger.record_tabular('Weightend', L.get_all_param_values(self._l_mean)[-4][3][0])

    @property
    def distribution(self):
        return self._dist
