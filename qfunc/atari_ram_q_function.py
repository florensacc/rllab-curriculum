import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import cgtcompat as theano
import cgtcompat.tensor as T
from .lasagne_q_function import LasagneQFunction
from core.serializable import Serializable

class AtariRAMQFunction(LasagneQFunction, Serializable):

    def __init__(self, mdp, hidden_sizes=[64], nonlinearity=NL.rectify):
        # create network
        input_var = T.matrix('input')
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=hidden_size,
                nonlinearity=nonlinearity,
                W=lasagne.init.HeUniform(),
                b=lasagne.init.Constant(.1),
                name="h%d" % idx
            )
        qval_layer = L.DenseLayer(
            l_hidden,
            num_units=mdp.n_actions,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.1),
            name="output_qval"
        )

        qval_var = L.get_output(qval_layer)

        self._n_actions = mdp.n_actions
        self._input_var = input_var
        self._qval_var = qval_var
        self._compute_qval = theano.function([input_var], qval_var, allow_input_downcast=True)

        super(AtariRAMQFunction, self).__init__([qval_layer])
        Serializable.__init__(self, mdp, hidden_sizes, nonlinearity)

    def compute_qval(self, states):
        return self._compute_qval(states)

    @property
    def qval_var(self):
        return self._qval_var

    @property
    def input_var(self):
        return self._input_var
