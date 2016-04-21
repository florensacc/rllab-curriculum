from __future__ import print_function
from __future__ import absolute_import

import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano.tensor as TT
import numpy as np
import itertools

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from sandbox.rocky.snn.core.network import StochasticMLP
from rllab.core.lasagne_helpers import get_full_output
from rllab.distributions.diagonal_gaussian import DiagonalGaussian


class StochasticGaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            input_latent_vars=None,
            hidden_sizes=(32, 32),
            hidden_latent_vars=None,
            learn_std=True,
            init_std=1.0,
            hidden_nonlinearity=NL.tanh,
            output_nonlinearity=None,
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        mean_network = StochasticMLP(
            input_shape=(obs_dim,),
            input_latent_vars=input_latent_vars,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            hidden_latent_vars=hidden_latent_vars,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_var

        l_log_std = ParamLayer(
            mean_network.input_layer,
            num_units=action_dim,
            param=lasagne.init.Constant(np.log(init_std)),
            name="output_log_std",
            trainable=learn_std,
        )

        self._mean_network = mean_network
        self._n_latent_layers = len(mean_network.latent_layers)
        self._l_mean = l_mean
        self._l_log_std = l_log_std

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(StochasticGaussianMLPPolicy, self).__init__(env_spec)

        outputs = self.dist_info_sym(mean_network.input_var)
        latent_keys = sorted(set(outputs.keys()).difference({"mean", "log_std"}))

        extras = get_full_output(
            [self._l_mean, self._l_log_std] + self._mean_network.latent_layers,
        )[1]
        latent_distributions = [extras[layer]["distribution"] for layer in self._mean_network.latent_layers]

        self._latent_keys = latent_keys
        self._latent_distributions = latent_distributions
        self._dist = DiagonalGaussian()

        self._f_dist_info = ext.compile_function(
            inputs=[obs_var],
            outputs=outputs,
        )

    def dist_info_sym(self, obs_var, state_info_vars=None):
        if state_info_vars is not None:
            latent_vars = {
                latent_layer: state_info_vars["latent_%d" % idx]
                for idx, latent_layer in enumerate(self._mean_network.latent_layers)
                }
        else:
            latent_vars = dict()
        all_outputs, extras = get_full_output(
            [self._l_mean, self._l_log_std] + self._mean_network.latent_layers,
            inputs={self._mean_network._l_in: obs_var},
            latent_givens=latent_vars,
        )

        mean_var = all_outputs[0]
        log_std_var = all_outputs[1]
        latent_vars = all_outputs[2:]
        latent_dist_infos = []
        for latent_layer in self._mean_network.latent_layers:
            latent_dist_infos.append(extras[latent_layer]["dist_info"])

        output_dict = dict(mean=mean_var, log_std=log_std_var)
        for idx, latent_var, latent_dist_info in zip(itertools.count(), latent_vars, latent_dist_infos):
            output_dict["latent_%d" % idx] = latent_var
            for k, v in latent_dist_info.iteritems():
                output_dict["latent_%d_%s" % (idx, k)] = v

        return output_dict

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        kl = self._dist.kl_sym(old_dist_info_vars, new_dist_info_vars)
        for idx, latent_dist in enumerate(self._latent_distributions):
            # collect dist info for each latent variable
            prefix = "latent_%d_" % idx
            old_latent_dist_info = {k[len(prefix):]: v for k, v in old_dist_info_vars.iteritems() if k.startswith(
                prefix)}
            new_latent_dist_info = {k[len(prefix):]: v for k, v in new_dist_info_vars.iteritems() if k.startswith(
                prefix)}
            kl += latent_dist.kl_sym(old_latent_dist_info, new_latent_dist_info)
        return kl

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        lr = self._dist.likelihood_ratio_sym(x_var, old_dist_info_vars, new_dist_info_vars)
        for idx, latent_dist in enumerate(self._latent_distributions):
            latent_var = old_dist_info_vars["latent_%d" % idx]
            prefix = "latent_%d_" % idx
            old_latent_dist_info = {k[len(prefix):]: v for k, v in old_dist_info_vars.iteritems() if k.startswith(
                prefix)}
            new_latent_dist_info = {k[len(prefix):]: v for k, v in new_dist_info_vars.iteritems() if k.startswith(
                prefix)}
            lr *= latent_dist.likelihood_ratio_sym(latent_var, old_latent_dist_info, new_latent_dist_info)
        return lr

    def entropy(self, dist_info):
        ent = self._dist.entropy(dist_info)
        for idx, latent_dist in enumerate(self._latent_distributions):
            prefix = "latent_%d_" % idx
            latent_dist_info = {k[len(prefix):]: v for k, v in dist_info.iteritems() if k.startswith(prefix)}
            ent += latent_dist.entropy(latent_dist_info)
        return ent

    @property
    def dist_info_keys(self):
        return ["mean", "log_std"] + self._latent_keys

    @overrides
    def get_action(self, observation):
        outputs = {k: v[0] for k, v in self._f_dist_info([observation]).iteritems()}
        mean = outputs["mean"]
        log_std = outputs["log_std"]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, outputs

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self

    @property
    def state_info_keys(self):
        return self._latent_keys
