import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import tensorfuse.tensor as TT
from pydoc import locate
from rllab.policy.base import DeterministicPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.ext import compile_function


class MeanNNPolicy(DeterministicPolicy, LasagnePowered, Serializable):
    """
    A policy that just outputs a mean (i.e. a deterministic policy)
    """

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('nonlinearity', type=str,
                  help='nonlinearity used for each hidden layer, can be one '
                       'of tanh, sigmoid')
    # pylint: disable=dangerous-default-value
    def __init__(self, mdp, hidden_sizes=[400, 300], nonlinearity=NL.rectify):
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
                W=lasagne.init.HeUniform(),
                name="h%d" % idx)
        output_layer = L.DenseLayer(
            l_hidden,
            num_units=mdp.action_dim,
            nonlinearity=NL.tanh,
            W=lasagne.init.Uniform(-3e-3, 3e-3),#Normal(0.01),
            name="output")

        actions_var = L.get_output(output_layer)

        self._output_layer = output_layer

        self._f_actions = compile_function([input_var], actions_var)

        super(MeanNNPolicy, self).__init__(mdp)
        LasagnePowered.__init__(self, [output_layer])
        Serializable.__init__(self, mdp, hidden_sizes, nonlinearity)

    @property
    @overrides
    def action_dim(self):
        return self._action_dim

    @property
    @overrides
    def action_dtype(self):
        return self._action_dtype

    @overrides
    def get_action_sym(self, input_var):
        return L.get_output(self._output_layer, input_var)

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_actions(self, observations):
        return self._f_actions(observations), [None] * len(observations)

    @overrides
    def get_action(self, observation):
        actions, pdists = self.get_actions([observation])
        return actions[0], pdists[0]
