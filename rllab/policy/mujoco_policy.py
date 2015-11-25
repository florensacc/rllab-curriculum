import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import tensorfuse as theano
import tensorfuse.tensor as TT
from pydoc import locate
from rllab.policy.lasagne_layers import ParamLayer
from rllab.policy.lasagne_policy import LasagnePolicy
from rllab.misc.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import autoargs


def log_normal_pdf(x, mean, log_std):
    normalized = (x - mean) / TT.exp(log_std)
    return -0.5*TT.square(normalized) - np.log((2*np.pi)**0.5) - log_std


class MujocoPolicy(LasagnePolicy, Serializable):

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    # pylint: disable=dangerous-default-value
    def __init__(self, mdp, hidden_sizes=[32, 32], nonlinearity=NL.tanh):
        # pylint: enable=dangerous-default-value
        # create network
        if isinstance(nonlinearity, str):
            nonlinearity = locate('lasagne.nonlinearities.' + nonlinearity)
        input_var = TT.matrix('input',
                              fixed_shape=(None, mdp.observation_shape[0]))
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
        mean_layer = L.DenseLayer(
            l_hidden,
            num_units=mdp.action_dim,
            nonlinearity=None,
            W=lasagne.init.Normal(0.01),
            name="output_mean")
        log_std_layer = ParamLayer(
            l_input,
            num_units=mdp.action_dim,
            param=lasagne.init.Constant(0),
            name="output_log_std")

        mean_var = L.get_output(mean_layer)
        log_std_var = L.get_output(log_std_layer)

        self._action_dim = mdp.action_dim
        self._input_var = input_var
        self._mean_var = mean_var
        self._log_std_var = log_std_var
        self._pdist_var = TT.concatenate([mean_var, log_std_var], axis=1)
        self._compute_action_params = theano.function(
            [input_var],
            [mean_var, log_std_var],
            allow_input_downcast=True
        )

        super(MujocoPolicy, self).__init__([mean_layer, log_std_layer])
        Serializable.__init__(self, mdp, hidden_sizes, nonlinearity)

    @property
    @overrides
    def pdist_var(self):
        return self._pdist_var

    @property
    @overrides
    def input_var(self):
        return self._input_var

    @overrides
    def new_action_var(self, name):
        return TT.matrix(name)

    # Computes D_KL(p_old || p_new)
    @overrides
    def kl(self, old_pdist_var, new_pdist_var):
        old_mean, old_log_std = self._split_pdist(old_pdist_var)
        new_mean, new_log_std = self._split_pdist(new_pdist_var)
        old_std = TT.exp(old_log_std)
        new_std = TT.exp(new_log_std)
        # mean: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = TT.square(old_mean - new_mean) + \
            TT.square(old_std) - TT.square(new_std)
        denominator = 2*TT.square(new_std) + 1e-8
        return TT.sum(
            numerator / denominator + new_log_std - old_log_std, axis=1)

    @overrides
    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        old_mean, old_log_std = self._split_pdist(old_pdist_var)
        new_mean, new_log_std = self._split_pdist(new_pdist_var)
        logli_new = log_normal_pdf(action_var, new_mean, new_log_std)
        logli_old = log_normal_pdf(action_var, old_mean, old_log_std)
        return TT.exp(TT.sum(logli_new - logli_old, axis=1))

    def _split_pdist(self, pdist):
        mean = pdist[:, :self._action_dim]
        log_std = pdist[:, self._action_dim:]
        return mean, log_std

    @overrides
    def compute_entropy(self, pdist):
        _, log_std = self._split_pdist(pdist)
        return np.mean(np.sum(log_std + np.log(np.sqrt(2*np.pi*np.e)), axis=1))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, observations):
        means, log_stds = self._compute_action_params(observations)
        # first get standard normal samples
        rnd = np.random.randn(*means.shape)
        pdists = np.concatenate([means, log_stds], axis=1)
        # transform back to the true distribution
        actions = rnd * np.exp(log_stds) + means
        return actions, pdists

    @overrides
    def get_action(self, observation):
        actions, pdists = self.get_actions([observation])
        return actions[0], pdists[0]

    def get_log_prob_sym(self, action_var):
        mean_var = self._mean_var
        log_std_var = self._log_std_var
        stdn = (action_var - mean_var)
        stdn /= TT.exp(log_std_var)
        return - TT.sum(log_std_var, axis=1) - \
            0.5*TT.sum(TT.square(stdn), axis=1) - \
            0.5*self._action_dim*np.log(2*np.pi)
