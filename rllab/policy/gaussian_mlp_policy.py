import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import numpy as np
import theano.tensor as TT

from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import MLP
from rllab.spaces import Box

from rllab.core.serializable import Serializable
from rllab.policy.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext
from rllab.misc import autoargs
from rllab.distributions.diagonal_gaussian import DiagonalGaussian


class GaussianMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('std_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected layers for std, note'
                       'there is a difference in semantics than above: here an empty'
                       'list means that std is independent of input and the last size is ignored')
    @autoargs.arg('initial_std', type=float,
                  help='Initial std')
    @autoargs.arg('std_trainable', type=bool,
                  help='Is std trainable')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to hidden layers')
    def __init__(
            self,
            env_spec,
            hidden_sizes=(32, 32),
            learn_std=True,
            init_std=1.0,
            adaptive_std=False,
            std_share_network=False,
            std_hidden_sizes=(32, 32),
            # We can't use None here since None is actually a valid value!
            std_nonlinearity=NL.rectify,
            nonlinearity=NL.rectify,
            output_nonlinearity=None,
    ):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Box)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        # create network
        mean_network = MLP(
            input_shape=(obs_dim,),
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            nonlinearity=nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )

        l_mean = mean_network.output_layer
        obs_var = mean_network.input_var

        if adaptive_std:
            l_log_std = MLP(
                input_shape=(obs_dim,),
                input_var=obs_var,
                output_dim=action_dim,
                hidden_sizes=std_hidden_sizes,
                nonlinearity=std_nonlinearity,
                output_nonlinearity=None,
            ).output_layer
        else:
            l_log_std = ParamLayer(
                mean_network.input_layer,
                num_units=action_dim,
                param=lasagne.init.Constant(np.log(init_std)),
                name="output_log_std",
                trainable=learn_std,
            )

        mean_var, log_std_var = L.get_output([l_mean, l_log_std])

        self._l_mean = l_mean
        self._l_log_std = l_log_std

        self._dist = DiagonalGaussian()

        LasagnePowered.__init__(self, [l_mean, l_log_std])
        super(GaussianMLPPolicy, self).__init__(env_spec)

        self._f_dist = ext.compile_function(
            inputs=[obs_var],
            outputs=[mean_var, log_std_var],
        )

    def dist_info_sym(self, obs_var, action_var):
        mean_var, log_std_var = L.get_output([self._l_mean, self._l_log_std], obs_var)
        return dict(mean=mean_var, log_std=log_std_var)

    # Computes D_KL(p_old || p_new)
    # @overrides
    # def kl_sym(self, dold_pdist_var, new_pdist_var):
    #     old_mean, old_log_std = self._split_pdist(old_pdist_var)
    #     new_mean, new_log_std = self._split_pdist(new_pdist_var)
    #     old_std = TT.exp(old_log_std)
    #     new_std = TT.exp(new_log_std)
    #     # mean: (N*A)
    #     # std: (N*A)
    #     # formula:
    #     # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
    #     # ln(\sigma_2/\sigma_1)
    #     numerator = TT.square(old_mean - new_mean) + \
    #                 TT.square(old_std) - TT.square(new_std)
    #     denominator = 2 * TT.square(new_std) + 1e-8
    #     return TT.sum(
    #         numerator / denominator + new_log_std - old_log_std, axis=1)
    #
    # @overrides
    # def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
    #     old_mean, old_log_std = self._split_pdist(old_pdist_var)
    #     new_mean, new_log_std = self._split_pdist(new_pdist_var)
    #     logli_new = log_normal_pdf(action_var, new_mean, new_log_std)
    #     logli_old = log_normal_pdf(action_var, old_mean, old_log_std)
    #     return TT.exp(TT.sum(logli_new - logli_old, axis=1))
    #
    # def _split_pdist(self, pdist):
    #     mean = pdist[:, :self.action_dim]
    #     log_std = pdist[:, self.action_dim:]
    #     return mean, log_std
    #
    # @overrides
    # def compute_entropy(self, pdist):
    #     _, log_std = self._split_pdist(pdist)
    #     return np.mean(np.sum(log_std + np.log(np.sqrt(2 * np.pi * np.e)), axis=1))

    # @property
    # @overrides
    # def pdist_dim(self):
    #     return self.action_dim * 2

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    # @overrides
    # def get_actions(self, observations):
    #     means, log_stds = self._compute_action_params(observations)
    #     # first get standard normal samples
    #     rnd = np.random.randn(*means.shape)
    #     pdists = np.concatenate([means, log_stds], axis=1)
    #     # transform back to the true distribution
    #     actions = rnd * np.exp(log_stds) + means
    #     return actions, pdists

    @overrides
    def get_action(self, observation):
        mean, log_std = [x[0] for x in self._f_dist([observation])]
        rnd = np.random.randn(len(mean))#*means.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)

    # def get_reparam_action_sym(self, obs_var, eta_var):
    #     means, log_stds = self._split_pdist(self.get_pdist_sym(obs_var))
    #     return eta_var * TT.exp(log_stds) + means
    #
    # def infer_eta(self, pdists, actions):
    #     means, log_stds = self._split_pdist(pdists)
    #     return (actions - means) / np.exp(log_stds)
    #
    # def get_log_prob_sym(self, input_var, action_var):
    #     mean_var = L.get_output(self._mean_layer, input_var)
    #     log_std_var = L.get_output(self._log_std_layer, input_var)
    #     stdn = (action_var - mean_var)
    #     stdn /= TT.exp(log_std_var)
    #     return - TT.sum(log_std_var, axis=1) - \
    #            0.5 * TT.sum(TT.square(stdn), axis=1) - \
    #            0.5 * self.action_dim * np.log(2 * np.pi)
    #
    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist
