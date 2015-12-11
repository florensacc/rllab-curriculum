from qfunc import LasagneQFunction
from core.serializable import Serializable
import theano
import theano.tensor as T
import lasagne.layers as L
import lasagne

class TabularQFunction(LasagneQFunction, Serializable):

    def __init__(self, mdp):
        input_var = T.matrix('input')
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]), input_var=input_var)
        l_output = L.DenseLayer(l_input, num_units=mdp.action_dim, nonlinearity=None)
        qval_var = L.get_output(l_output)

        self._qval_var = qval_var
        self._compute_qval = theano.function([input_var], qval_var, allow_input_downcast=True)
        self._input_var = input_var
        self._action_dim = mdp.n_actions
        super(TabularQFunction, self).__init__([l_output])
        Serializable.__init__(self, mdp)

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def input_var(self):
        return self._input_var

    @property
    def qval_var(self):
        return self._qval_var

    def compute_qval(self, observations):
        return self._compute_qval(observations)
