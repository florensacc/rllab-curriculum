from .lasagne_policy import LasagnePolicy
from core.serializable import Serializable
from misc.special import weighted_sample
from misc.overrides import overrides
import numpy as np
import tensorfuse as theano
import tensorfuse.tensor as T
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne

class TabularPolicy(LasagnePolicy, Serializable):

    def __init__(self, mdp):
        input_var = T.matrix('input')
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]), input_var=input_var)
        l_output = L.DenseLayer(l_input, num_units=mdp.n_actions, nonlinearity=NL.softmax)
        prob_var = L.get_output(l_output)

        self._pdist_var = prob_var
        self._compute_probs = theano.function([input_var], prob_var, allow_input_downcast=True)
        self._input_var = input_var
        self._n_actions = mdp.n_actions
        super(TabularPolicy, self).__init__([l_output])
        Serializable.__init__(self, mdp)

    @property
    def n_actions(self):
        return self._n_actions

    @property
    @overrides
    def input_var(self):
        return self._input_var

    @property
    @overrides
    def pdist_var(self):
        return self._pdist_var

    @overrides
    def new_action_var(self, name):
        return T.imatrix(name)

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return T.sum(old_prob_var * (T.log(old_prob_var) - T.log(new_prob_var)), axis=1)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        N = old_prob_var.shape[0]
        return new_prob_var[T.arange(N), T.reshape(action_var, (-1,))] / old_prob_var[T.arange(N), T.reshape(action_var, (-1,))]

    @overrides
    def compute_entropy(self, prob):
        return -np.mean(np.sum(prob * np.log(prob), axis=1))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, states):
        probs = self._compute_probs(states)
        actions = [weighted_sample(prob, range(len(prob))) for prob in probs]
        return actions, probs
