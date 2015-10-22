import os
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
from lasagne_layers import ParamLayer, OpLayer
from lasagne_policy import LasagnePolicy
import numpy as np
import cgtcompat as theano
import cgtcompat.tensor as T
from core.serializable import Serializable

def normal_pdf(x, mean, log_std):
    return T.exp(-T.square((x - mean) / T.exp(log_std)) / 2) / ((2*np.pi)**0.5 * T.exp(log_std))

def log_normal_pdf(x, mean, log_std):
    normalized = (x - mean) / T.exp(log_std)
    return -0.5*T.square(normalized) - np.log((2*np.pi)**0.5) - log_std

class MujocoPolicy(LasagnePolicy, Serializable):

    def __init__(self, mdp, hidden_sizes=[32,32], nonlinearity=NL.tanh):

        # create network
        input_var = T.matrix('input')
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=hidden_size, nonlinearity=nonlinearity, W=lasagne.init.Normal(0.1), name="h%d" % idx)
        mean_layer = L.DenseLayer(l_hidden, num_units=mdp.n_actions, nonlinearity=None, W=lasagne.init.Normal(0.01), name="output_mean")
        log_std_layer = ParamLayer(l_input, num_units=mdp.n_actions, param=lasagne.init.Constant(0.), name="output_log_std")

        mean_var = L.get_output(mean_layer)
        log_std_var = L.get_output(log_std_layer)

        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.observation_shape = mdp.observation_shape
        self.n_actions = mdp.n_actions
        self.input_var = input_var
        self.pdist_var = T.concatenate([mean_var, log_std_var], axis=1)
        self.compute_action_params = theano.function([input_var], [mean_var, log_std_var], allow_input_downcast=True)

        super(MujocoPolicy, self).__init__([mean_layer, log_std_layer])
        Serializable.__init__(self, mdp, hidden_sizes, nonlinearity)

    def kl(self, old_pdist_var, new_pdist_var):
        old_mean, old_log_std = self.split_pdist(old_pdist_var)
        new_mean, new_log_std = self.split_pdist(new_pdist_var)
        old_std = T.exp(old_log_std)
        new_std = T.exp(new_log_std)
        # mean: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_262) + ln(\sigma_2/\sigma_1)
        return T.sum((T.square(old_mean - new_mean) + T.square(old_std) - T.square(new_std)) / (2*T.square(new_std) + 1e-8) + new_log_std - old_log_std, axis=1)

    def likelihood_ratio(self, old_pdist_var, new_pdist_var, action_var):
        old_mean, old_log_std = self.split_pdist(old_pdist_var)
        new_mean, new_log_std = self.split_pdist(new_pdist_var)
        logli_new = log_normal_pdf(action_var, new_mean, new_log_std)
        logli_old = log_normal_pdf(action_var, old_mean, old_log_std)
        return T.exp(T.sum(logli_new - logli_old, axis=1))

    def split_pdist(self, pdist):
        mean = pdist[:, :self.n_actions]
        log_std = pdist[:, self.n_actions:]
        return mean, log_std

    def compute_entropy(self, pdist):
        mean, log_std = self.split_pdist(pdist)
        return np.mean(np.sum(log_std + np.log(np.sqrt(2*np.pi*np.e)), axis=1))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_actions(self, states):
        means, log_stds = self.compute_action_params(states)
        # first get standard normal samples
        rnd = np.random.randn(*means.shape)
        pdists = np.concatenate([means, log_stds], axis=1)
        # transform back to the true distribution
        actions = rnd * np.exp(log_stds) + means
        return actions, pdists

    def get_action(self, state):
        actions, pdists = self.get_actions([state])
        return actions[0], pdists[0]
