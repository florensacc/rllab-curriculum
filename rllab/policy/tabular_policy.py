from rllab.policy.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc.special import weighted_sample
from rllab.misc.overrides import overrides
from rllab.misc.ext import compile_function
import numpy as np
import tensorfuse.tensor as TT
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL


class TabularPolicy(StochasticPolicy, LasagnePowered, Serializable):

    def __init__(self, mdp):
        input_var = TT.matrix('input')
        l_input = L.InputLayer(
            shape=(None, mdp.observation_shape[0]),
            input_var=input_var)

        l_output = L.DenseLayer(l_input,
                                num_units=mdp.action_dim,
                                W=lasagne.init.Constant(0.),
                                b=None,
                                nonlinearity=NL.softmax)

        prob_var = L.get_output(l_output)

        self._output_layer = l_output
        self._f_probs = compile_function([input_var], prob_var)
        super(TabularPolicy, self).__init__(mdp)
        LasagnePowered.__init__(self, [l_output])
        Serializable.__init__(self, mdp)

    @property
    def action_dim(self):
        return self._action_dim

    @overrides
    def get_pdist_sym(self, input_var):
        return L.get_output(self._output_layer, input_var)

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return TT.sum(old_prob_var *
                      (TT.log(old_prob_var) - TT.log(new_prob_var)), axis=1)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        N = old_prob_var.shape[0]
        new_ll = new_prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]
        old_ll = old_prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]
        return new_ll / old_ll

    @overrides
    def compute_entropy(self, prob):
        return -np.mean(np.sum(prob * np.log(prob), axis=1))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, states):
        probs = self._f_probs(states)
        actions = [weighted_sample(prob, range(len(prob))) for prob in probs]
        return actions, probs

    @overrides
    def get_log_prob_sym(self, input_var, action_var):
        N = action_var.shape[0]
        prob_var = L.get_output(self._output_layer, input_var)
        prob = prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]
        return TT.log(prob)
