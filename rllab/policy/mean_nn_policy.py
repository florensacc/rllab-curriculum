import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import theano.tensor as TT
import itertools
from pydoc import locate
from rllab.policy.base import DeterministicPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.lasagne_layers import batch_norm
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.ext import compile_function, new_tensor


class MeanNNPolicy(DeterministicPolicy, LasagnePowered, Serializable):
    """
    A policy that just outputs a mean (i.e. a deterministic policy)
    """

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('hidden_nl', type=str, nargs='*',
                  help='list of nonlinearities for the hidden layers')
    @autoargs.arg('hidden_W_init', type=str, nargs='*',
                  help='list of initializers for W for the hidden layers')
    @autoargs.arg('hidden_b_init', type=str, nargs='*',
                  help='list of initializers for b for the hidden layers')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('output_W_init', type=str,
                  help='initializer for W for the output layer')
    @autoargs.arg('output_b_init', type=str,
                  help='initializer for b for the output layer')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to all layers')
    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            mdp,
            hidden_sizes=[100, 100],
            hidden_nl=['lasagne.nonlinearities.rectify'],
            hidden_W_init=['lasagne.init.HeUniform()'],
            hidden_b_init=['lasagne.init.Constant(0.)'],
            output_nl='None',
            output_W_init='lasagne.init.Uniform(-3e-3, 3e-3)',
            output_b_init='lasagne.init.Uniform(-3e-3, 3e-3)',
            bn=False):
        # pylint: enable=dangerous-default-value
        # create network
        input_var = TT.matrix('input')
                              
        l_input = L.InputLayer(shape=(None, mdp.observation_shape[0]),
                               input_var=input_var)

        if len(hidden_nl) == 1:
            hidden_nl *= len(hidden_sizes)
        assert len(hidden_nl) == len(hidden_sizes)

        if len(hidden_W_init) == 1:
            hidden_W_init *= len(hidden_sizes)
        assert len(hidden_W_init) == len(hidden_sizes)

        if len(hidden_b_init) == 1:
            hidden_b_init *= len(hidden_sizes)
        assert len(hidden_b_init) == len(hidden_sizes)

        l_hidden = l_input

        for idx, size, nl, W_init, b_init in zip(
                itertools.count(), hidden_sizes, hidden_nl,
                hidden_W_init, hidden_b_init):
            l_hidden = L.DenseLayer(
                l_hidden,
                num_units=size,
                W=eval(W_init),
                b=eval(b_init),
                nonlinearity=eval(nl),
                name="h%d" % idx
            )
            if bn:
                l_hidden = batch_norm(l_hidden)

        l_output = L.DenseLayer(
            l_hidden,
            num_units=mdp.action_dim,
            W=eval(output_W_init),
            b=eval(output_b_init),
            nonlinearity=eval(output_nl),
            name="output"
        )

        # if bn:
        #     l_output = batch_norm(l_output)

        action_var = L.get_output(l_output)

        self._output_layer = l_output

        self._f_actions = compile_function([input_var], action_var)

        super(MeanNNPolicy, self).__init__(mdp)
        LasagnePowered.__init__(self, [l_output])
        Serializable.__init__(
            self, mdp=mdp, hidden_sizes=hidden_sizes, hidden_nl=hidden_nl,
            hidden_W_init=hidden_W_init, hidden_b_init=hidden_b_init,
            output_nl=output_nl, output_W_init=output_W_init,
            output_b_init=output_b_init, bn=bn)

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
