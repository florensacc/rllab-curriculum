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
from misc.overrides import overrides

def weighted_sample(weights, objects):
    """Return a random item from objects, with the weighting defined by weights 
    (which must sum to 1)."""
    cs = np.cumsum(weights) #An array of the weights, cumulatively summed.
    idx = sum(cs < np.random.rand()) #Find the index of the first weight over a random value.
    return objects[idx]

class AtariRAMPolicy(LasagnePolicy, Serializable):

    def __init__(self, mdp, hidden_sizes=[64], nonlinearity=NL.tanh):

        # create network
        input_var = T.matrix('input')
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=hidden_size, nonlinearity=nonlinearity, W=lasagne.init.Normal(0.1), name="h%d" % idx)
        prob_layer = L.DenseLayer(l_hidden, num_units=mdp.n_actions, nonlinearity=NL.softmax, W=lasagne.init.Normal(0.01), name="output_prob")

        prob_var = L.get_output(prob_layer)

        self._n_actions = mdp.n_actions
        self._input_var = input_var
        self._pdist_var = prob_var
        self._compute_action_params = theano.function([input_var], prob_var, allow_input_downcast=True)

        super(AtariRAMPolicy, self).__init__([prob_layer])
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
        return T.imatrix(name)

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return T.sum(old_prob_var * (T.log(old_prob_var) - T.log(new_prob_var)), axis=1)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        N = old_prob_var.shape[0]
        return new_prob_var[T.arange(N), action_var.reshape((-1,))] / old_prob_var[T.arange(N), action_var.reshape((-1,))]

    @overrides
    def compute_entropy(self, prob):
        return -np.mean(np.sum(prob * np.log(prob), axis=1))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, states):
        probs = self._compute_action_params(states)
        actions = [weighted_sample(prob, range(len(prob))) for prob in probs]
        return actions, probs

    @overrides
    def get_action(self, state):
        actions, probs = self.get_actions([state])
        return actions[0], probs[0]
