import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import tensorfuse as theano
import tensorfuse.tensor as T
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc.special import weighted_sample
from rllab.policy.base import StochasticPolicy


class AtariRAMPolicy(StochasticPolicy, LasagnePowered, Serializable):

    def __init__(
            self,
            mdp,
            hidden_sizes=[64],
            nonlinearity=NL.tanh):
        # create network
        input_var = T.matrix('input')
        l_input = L.InputLayer(
            shape=(None, mdp.observation_shape[0]),
            input_var=input_var
        )
        l_hidden = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=hidden_size,
                nonlinearity=nonlinearity,
                W=lasagne.init.Normal(0.1),
                name="h%d" % idx
            )
        prob_layer = L.DenseLayer(
            l_hidden,
            num_units=mdp.action_dim,
            nonlinearity=NL.softmax,
            W=lasagne.init.Normal(0.01),
            name="output_prob"
        )

        prob_var = L.get_output(prob_layer)

        self._prob_layer = prob_layer

        self._compute_probs = theano.function([input_var], prob_var, allow_input_downcast=True)

        super(AtariRAMPolicy, self).__init__(mdp)
        LasagnePowered.__init__(self, [prob_layer])
        Serializable.__init__(self, mdp, hidden_sizes, nonlinearity)

    @overrides
    def get_pdist_sym(self, input_var):
        return L.get_output(self._prob_layer, input_var)

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
        probs = self._compute_probs(states)
        actions = [weighted_sample(prob, range(len(prob))) for prob in probs]
        return actions, probs
