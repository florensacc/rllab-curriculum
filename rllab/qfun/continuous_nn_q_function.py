import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import tensorfuse.tensor as TT

from rllab.qfun.base import ContinuousQFunction
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable


class ContinuousNNQFunction(ContinuousQFunction, LasagnePowered, Serializable):

    # @autoargs.arg('hidden_sizes', type=int, nargs='*',
    #               help='list of sizes for the fully-connected hidden layers')
    # @autoargs.arg('nonlinearity', type=str,
    #               help='nonlinearity used for each hidden layer, can be one '
    #                    'of tanh, sigmoid')
    # pylint: disable=dangerous-default-value
    def __init__(self, mdp):  # , hidden_sizes=[32, 32], nonlinearity=NL.tanh):
        # pylint: enable=dangerous-default-value
        # create network
        # if isinstance(nonlinearity, str):
        #     nonlinearity = locate('lasagne.nonlinearities.' + nonlinearity)
        obs_var = TT.tensor(
            'obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        action_var = TT.matrix(
            'action',
            dtype=mdp.action_dtype
        )
        l_obs = L.InputLayer(shape=(None,) + mdp.observation_shape,
                             input_var=obs_var)
        l_action = L.InputLayer(shape=(None, mdp.action_dim),
                                input_var=action_var)
        l_hidden1 = L.DenseLayer(
            l_obs,
            num_units=400,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(0.),
            nonlinearity=NL.rectify,
            name="h1"
        )
        l_hidden2 = L.DenseLayer(
            l_hidden1,
            num_units=300,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(0.),
            nonlinearity=NL.rectify,
            name="h2"
        )

        l_with_action = L.ConcatLayer([l_hidden2, l_action])

        l_output = L.DenseLayer(
            l_with_action,
            num_units=1,
            W=lasagne.init.Uniform(-3e-3, 3e-3),
            b=lasagne.init.Uniform(-3e-3, 3e-3),
            nonlinearity=None,
            name="output"
        )

        self._output_layer = l_output
        self._obs_layer = l_obs
        self._action_layer = l_action

        super(ContinuousNNQFunction, self).__init__(mdp)
        LasagnePowered.__init__(self, [l_output])
        Serializable.__init__(self, mdp)

    def get_qval_sym(self, obs_var, action_var):
        qvals = L.get_output(
            self._output_layer,
            {self._obs_layer: obs_var, self._action_layer: action_var}
        )
        return TT.reshape(qvals, (-1,))
