import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import theano
import theano.tensor as TT
from pydoc import locate
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.policy.base import StochasticPolicy
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.special import weighted_sample, softmax, softmax_sym
from rllab.misc.ext import new_tensor


def normal_pdf_sym(x, mean, log_std):
    std = TT.exp(log_std)
    normalized = (x - mean) / std
    return TT.prod(
        TT.exp(
            -0.5*TT.square(normalized) - np.log((2*np.pi)**0.5) - log_std
        ),
        axis=1
    )


def gmm_pdf_sym(x, means, log_stds, log_mixture_weights, n_mixtures):
    mixture_weights = softmax_sym(log_mixture_weights)
    result = 0
    for k in range(n_mixtures):
        w = mixture_weights[:, k]
        mean = means[:, k]
        log_std = log_stds[:, k]
        result += w * normal_pdf_sym(x, mean, log_std)
    return result


class GMMNNPolicy(StochasticPolicy, LasagnePowered, Serializable):
    """
    A mixture-of-Gaussian policy. Different from a Gaussian policy which just
    outputs a mean and a standard deviation, this policy outputs K mixtures
    of Gaussian distributions, allowing for potentially more diverse
    exploration, and less susceptible to local optima.
    """

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('n_mixtures', type=int,
                  help='Number of mixtures.')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    @autoargs.arg('lr_approx_mode', type=str, choices=['markov', 'jensen'],
                  help='Approximation mode for the likelihood ratio.')
    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            mdp,
            hidden_sizes=[32, 32],
            n_mixtures=5,
            nonlinearity=NL.tanh,
            lr_approx_mode='markov'):
        # pylint: enable=dangerous-default-value
        # create network
        if isinstance(nonlinearity, str):
            nonlinearity = locate('lasagne.nonlinearities.' + nonlinearity)
        input_var = new_tensor(
            'input',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]),
                               input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=hidden_size,
                nonlinearity=nonlinearity,
                W=lasagne.init.Normal(0.1),
                name="h%d" % idx)
        means_layer = L.DenseLayer(
            l_hidden,
            num_units=(mdp.action_dim * n_mixtures),
            nonlinearity=None,
            W=lasagne.init.Normal(0.01),
            name="output_means")
        log_stds_layer = ParamLayer(
            l_input,
            num_units=(mdp.action_dim * n_mixtures),
            param=lasagne.init.Constant(0.),
            name="output_log_std")
        log_mixture_weights_layer = ParamLayer(
            l_input,
            num_units=n_mixtures,
            param=lasagne.init.Constant(0.),
            name="output_log_mixture_weights")

        if lr_approx_mode not in ['markov', 'jensen']:
            raise ValueError('Unrecognized lr_approx_mode!')

        self._lr_approx_mode = lr_approx_mode
        self._n_mixtures = n_mixtures
        self._means_layer = means_layer
        self._log_stds_layer = log_stds_layer
        self._log_mixture_weights_layer = log_mixture_weights_layer
        self._compute_pdists = theano.function(
            [input_var],
            self.get_pdist_sym(input_var),
            allow_input_downcast=True
        )
        self._mixture_id = None

        super(GMMNNPolicy, self).__init__(mdp)
        LasagnePowered.__init__(
            self, [means_layer, log_stds_layer, log_mixture_weights_layer])
        Serializable.__init__(
            self, mdp, hidden_sizes, n_mixtures, nonlinearity)

    def get_pdist_sym(self, input_var):
        means_var = L.get_output(self._means_layer, input_var)
        log_stds_var = L.get_output(self._log_stds_layer, input_var)
        log_mixture_weights_var = L.get_output(
            self._log_mixture_weights_layer, input_var)
        return TT.concatenate(
            [means_var, log_stds_var, log_mixture_weights_var], axis=1)

    # Computes D_KL(p_old || p_new)
    @overrides
    def kl(self, old_pdist_var, new_pdist_var):
        # For GMM, we cannot compute the exact KL divergence.
        # Instead, we resort to the following approximation based on the
        # log-sum inequality:
        # D_KL(old || new) <= D_KL(w_old || w_new) +
        #                       sum w_i D_KL(old_i || new_i)
        old_means, old_log_stds, old_log_mixture_weights = \
            self._split_pdist(old_pdist_var)
        new_means, new_log_stds, new_log_mixture_weights = \
            self._split_pdist(new_pdist_var)
        old_stds = TT.exp(old_log_stds)
        new_stds = TT.exp(new_log_stds)
        old_mixture_weights = softmax_sym(old_log_mixture_weights)
        new_mixture_weights = softmax_sym(new_log_mixture_weights)

        # First, add the contribution of D_KL(w_old || w_new), which is the KL
        # divergence of two categorical distributions
        result = TT.sum(
            old_mixture_weights *
            (TT.log(old_mixture_weights) - TT.log(new_mixture_weights)),
            axis=1)

        # mean: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        for i in range(self._n_mixtures):
            old_mean = old_means[:, i]
            new_mean = new_means[:, i]
            old_std = old_stds[:, i]
            new_std = new_stds[:, i]
            old_log_std = old_log_stds[:, i]
            new_log_std = new_log_stds[:, i]
            numerator = TT.square(old_mean - new_mean) + \
                TT.square(old_std) - TT.square(new_std)
            denominator = 2*TT.square(new_std) + 1e-8
            result += old_mixture_weights[:, i] * TT.sum(
                numerator / denominator + new_log_std - old_log_std, axis=1)
        return result

    @overrides
    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        old_means, old_log_stds, old_log_mixture_weights = \
            self._split_pdist(old_pdist_var)
        new_means, new_log_stds, new_log_mixture_weights = \
            self._split_pdist(new_pdist_var)
        if self._lr_approx_mode == 'markov':
            li_new = gmm_pdf_sym(
                action_var, new_means, new_log_stds, new_log_mixture_weights,
                self._n_mixtures)
            li_old = gmm_pdf_sym(
                action_var, old_means, old_log_stds, old_log_mixture_weights,
                self._n_mixtures)
            return li_new / li_old
        elif self._lr_approx_mode == 'jensen':
            ans = 0
            mixture_weights = softmax_sym(new_log_mixture_weights)
            for k in range(self._n_mixtures):
                new_mean = new_means[:, k]
                old_mean = old_means[:, k]
                new_log_std = new_log_stds[:, k]
                old_log_std = old_log_stds[:, k]
                new_pdf = normal_pdf_sym(action_var, new_mean, new_log_std)
                old_pdf = normal_pdf_sym(action_var, old_mean, old_log_std)
                w = mixture_weights[:, k]
                ans += w * new_pdf / old_pdf
            return ans
        else:
            assert False

    def _split_pdist(self, pdist):
        nw = self.action_dim * self._n_mixtures
        means = pdist[:, :nw].reshape(
            (pdist.shape[0], self._n_mixtures, self.action_dim)
        )
        log_stds = pdist[:, nw:nw*2].reshape(
            (pdist.shape[0], self._n_mixtures, self.action_dim)
        )
        log_mixture_weights = pdist[:, nw*2:]
        return means, log_stds, log_mixture_weights

    @overrides
    def compute_entropy(self, pdist):
        # For GMM, we cannot compute the exact entropy.
        # We use the following crude approximation, applying the concavity
        # of entropy:
        # H(sum w_i N(mu_i, sigma_i)) >= sum w_i H(N(mu_i, sigma_i))
        _, log_stds, log_mixture_weights = self._split_pdist(pdist)
        mixture_weights = softmax(log_mixture_weights)
        ent = 0
        for k in range(self._n_mixtures):
            log_std = log_stds[:, k]
            w = mixture_weights[:, k]
            ent += np.mean(
                w * np.sum(log_std + np.log(np.sqrt(2*np.pi*np.e)), axis=1))
        return ent

    def episode_reset(self):
        log_mixture_weights = \
            self._log_mixture_weights_layer.param.get_value(borrow=True)
        mixture_weights = softmax(
            log_mixture_weights.reshape((1, -1))).reshape((-1,))
        self._mixture_id = weighted_sample(
            mixture_weights, range(self._n_mixtures))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, observations):
        pdists = self._compute_pdists(observations)
        mixture_means, mixture_log_stds, _ = \
            self._split_pdist(pdists)
        # find the corresponding Gaussian distributions
        means = mixture_means[:, self._mixture_id]
        log_stds = mixture_log_stds[:, self._mixture_id]
        # get standard normal samples
        rnd = np.random.randn(*means.shape)
        # transform back to the true distribution
        actions = rnd * np.exp(log_stds) + means
        return actions, pdists

    @overrides
    def get_action(self, observation):
        actions, pdists = self.get_actions([observation])
        return actions[0], pdists[0]

    def get_log_prob_sym(self, input_var, action_var):
        means, log_stds, log_mixture_weights = \
            self._split_pdist(self.get_pdist_sym(input_var))
        return TT.log(
            gmm_pdf_sym(
                action_var, means, log_stds, log_mixture_weights,
                self._n_mixtures
            )
        )
