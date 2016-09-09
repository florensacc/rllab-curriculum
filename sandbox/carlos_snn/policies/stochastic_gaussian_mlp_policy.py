


import lasagne
import lasagne.nonlinearities as NL
import numpy as np
import itertools

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from sandbox.rocky.snn.core.network import StochasticMLP
from rllab.core.lasagne_helpers import get_full_output
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
import theano.tensor as TT


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
        obs_var = mean_network.input_layer.input_var

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
        self._dist = DiagonalGaussian(action_dim)

        self._f_dist_info = ext.compile_function(
            inputs=[obs_var],
            outputs=outputs,
        )
        self._f_dist_info_givens = None

    @property
    def latent_layers(self):
        return self._mean_network.latent_layers

    @property
    def latent_dims(self):
        return self._mean_network.latent_dims

    def dist_info(self, obs, state_infos=None):
        if state_infos is None or len(state_infos) == 0:
            return self._f_dist_info(obs)
        if self._f_dist_info_givens is None:
            # compile function
            obs_var = self._mean_network.input_var
            latent_keys = ["latent_%d" % idx for idx in range(self._n_latent_layers)]
            latent_vars = [TT.matrix("latent_%d" % idx) for idx in range(self._n_latent_layers)]
            latent_dict = dict(list(zip(latent_keys, latent_vars)))
            self._f_dist_info_givens = ext.compile_function(
                inputs=[obs_var] + latent_vars,
                outputs=self.dist_info_sym(obs_var, latent_dict),
            )
        latent_vals = []
        for idx in range(self._n_latent_layers):
            latent_vals.append(state_infos["latent_%d" % idx])
        return self._f_dist_info_givens(*[obs] + latent_vals)

    def reset(self):  #here I would sample a latents var.
        # sample latents
        # store it in self.something that then goes to all the others
        pass


    def dist_info_sym(self, obs_var, state_info_vars=None):
        if state_info_vars is not None:
            latent_givens = {
                latent_layer: state_info_vars["latent_%d" % idx]
                for idx, latent_layer in enumerate(self._mean_network.latent_layers)
                }
            latent_dist_infos = dict()
            for idx, latent_layer in enumerate(self._mean_network.latent_layers):
                cur_dist_info = dict()
                prefix = "latent_%d_" % idx
                for k, v in state_info_vars.items():
                    if k.startswith(prefix):
                        cur_dist_info[k[len(prefix):]] = v
                latent_dist_infos[latent_layer] = cur_dist_info
        else:
            latent_givens = dict()
            latent_dist_infos = dict()
        all_outputs, extras = get_full_output(
            [self._l_mean, self._l_log_std] + self._mean_network.latent_layers,
            inputs={self._mean_network._l_in: obs_var},
            latent_givens=latent_givens,
            latent_dist_infos=latent_dist_infos,
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
            for k, v in latent_dist_info.items():
                output_dict["latent_%d_%s" % (idx, k)] = v

        return output_dict

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of distributions of both the actions and the latents variables
        """
        kl = self._dist.kl_sym(old_dist_info_vars, new_dist_info_vars)
        for idx, latent_dist in enumerate(self._latent_distributions):
            # collect dist info for each latents variable
            prefix = "latent_%d_" % idx
            old_latent_dist_info = {k[len(prefix):]: v for k, v in old_dist_info_vars.items() if k.startswith(
                prefix)}
            new_latent_dist_info = {k[len(prefix):]: v for k, v in new_dist_info_vars.items() if k.startswith(
                prefix)}
            kl += latent_dist.kl_sym(old_latent_dist_info, new_latent_dist_info)
        return kl

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic likelihood ratio of both the actions and the latents variables.
        """
        lr = self._dist.likelihood_ratio_sym(action_var, old_dist_info_vars, new_dist_info_vars)
        for idx, latent_dist in enumerate(self._latent_distributions):
            latent_var = old_dist_info_vars["latent_%d" % idx]
            prefix = "latent_%d_" % idx
            old_latent_dist_info = {k[len(prefix):]: v for k, v in old_dist_info_vars.items() if k.startswith(
                prefix)}
            new_latent_dist_info = {k[len(prefix):]: v for k, v in new_dist_info_vars.items() if k.startswith(
                prefix)}
            lr *= latent_dist.likelihood_ratio_sym(latent_var, old_latent_dist_info, new_latent_dist_info)
        return lr

    def log_likelihood(self, actions, dist_info, action_only=False):
        """
        Computes the log likelihood of both the actions and the latents variables, unless action_only is set to True,
        in which case it will only compute the log likelihood of the actions.
        :return:
        """
        logli = self._dist.log_likelihood(actions, dist_info)
        if not action_only:
            for idx, latent_dist in enumerate(self._latent_distributions):
                latent_var = dist_info["latent_%d" % idx]
                prefix = "latent_%d_" % idx
                latent_dist_info = {k[len(prefix):]: v for k, v in dist_info.items() if k.startswith(
                    prefix)}
                logli += latent_dist.log_likelihood(latent_var, latent_dist_info)
        return logli

    def log_likelihood_sym(self, action_var, dist_info_vars):
        logli = self._dist.log_likelihood_sym(action_var, dist_info_vars)
        for idx, latent_dist in enumerate(self._latent_distributions):
            latent_var = dist_info_vars["latent_%d" % idx]
            prefix = "latent_%d_" % idx
            latent_dist_info = {k[len(prefix):]: v for k, v in dist_info_vars.items() if k.startswith(
                prefix)}
            logli += latent_dist.log_likelihood_sym(latent_var, latent_dist_info)
        return logli

    def entropy(self, dist_info):
        ent = self._dist.entropy(dist_info)
        for idx, latent_dist in enumerate(self._latent_distributions):
            prefix = "latent_%d_" % idx
            latent_dist_info = {k[len(prefix):]: v for k, v in dist_info.items() if k.startswith(prefix)}
            ent += latent_dist.entropy(latent_dist_info)
        return ent

    @property
    def dist_info_keys(self):
        return ["mean", "log_std"] + self._latent_keys

    @overrides
    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        outputs = self._f_dist_info(observations)
        mean = outputs["mean"]
        log_std = outputs["log_std"]
        rnd = np.random.normal(size=mean.shape)
        actions = rnd * np.exp(log_std) + mean
        return actions, outputs

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        """
        We set the distribution to the policy itself since we need some behavior different from a usual diagonal
        Gaussian distribution.
        """
        return self

    @property
    def state_info_keys(self):
        return self._latent_keys
