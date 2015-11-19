from .lasagne_policy import LasagnePolicy
from rllab.misc.serializable import Serializable
from rllab.misc.special import weighted_sample
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
import numpy as np
import tensorfuse as theano
import tensorfuse.tensor as TT
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne

class TabularPolicy(LasagnePolicy, Serializable):

    @autoargs.arg('init_weights', type=str, help='Distribution for initializing the weights. Default to uniform.')
    def __init__(self, mdp, init_weights='uniform'):
        input_var = TT.matrix('input')
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]), input_var=input_var)

        if init_weights == 'uniform':
            init_W = lasagne.init.Constant(0.)
            init_b = lasagne.init.Constant(0.)
        else:
            init_W = lasagne.init.GlorotUniform()
            init_b = lasagne.init.Constant(0.)

        l_output = L.DenseLayer(l_input, num_units=mdp.action_dim, W=init_W, b=init_b, nonlinearity=NL.softmax)
        prob_var = L.get_output(l_output)

        self._pdist_var = prob_var
        self._prob_var = prob_var
        self._compute_probs = theano.function([input_var], prob_var, allow_input_downcast=True)
        self._input_var = input_var
        self._action_dim = mdp.action_dim
        super(TabularPolicy, self).__init__([l_output])
        Serializable.__init__(self, mdp)

    @property
    def action_dim(self):
        return self._action_dim

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
        return TT.imatrix(name)

    @overrides
    def kl(self, old_prob_var, new_prob_var):
        return TT.sum(old_prob_var * (TT.log(old_prob_var) - TT.log(new_prob_var)), axis=1)

    @overrides
    def likelihood_ratio(self, old_prob_var, new_prob_var, action_var):
        N = old_prob_var.shape[0]
        return new_prob_var[TT.arange(N), TT.reshape(action_var, (-1,))] / old_prob_var[TT.arange(N), TT.reshape(action_var, (-1,))]

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

    @overrides
    def get_log_prob_sym(self, action_var):
        N = action_var.shape[0]
        return TT.log(self._prob_var[TT.arange(N), action_var])
