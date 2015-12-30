import theano.tensor as TT
from keras.models import Graph
from keras.layers.core import Dense, Layer
from keras.layers.normalization import BatchNormalization
from rllab.qf.base import ContinuousQFunction, NormalizableQFunction
from rllab.core.keras_powered import KerasPowered
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class ContinuousNNKerasQFunction(ContinuousQFunction, KerasPowered,
                                 NormalizableQFunction, Serializable):
    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('hidden_nl', type=str, nargs='*',
                  help='list of nonlinearities for the hidden layers')
    @autoargs.arg('hidden_init', type=str, nargs='*',
                  help='list of initializers for the hidden layers weights')
    @autoargs.arg('action_merge_layer', type=int,
                  help='Index for the hidden layer that the action kicks in')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('output_init', type=str,
                  help='initializer for the output layer weights')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to all layers')
    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            mdp,
            hidden_sizes=[100, 100],
            hidden_nl='relu',
            hidden_init='he_uniform',
            action_merge_layer=-2,
            output_nl='linear',
            output_init='he_uniform',
            bn=False):
        # pylint: enable=dangerous-default-value

        assert len(mdp.observation_shape) == 1

        graph = Graph()

        graph.add_input("observation", input_shape=mdp.observation_shape)
        graph.add_input("action", input_shape=(mdp.action_dim,))

        n_layers = len(hidden_sizes) + 1

        action_merge_layer = \
            (action_merge_layer % n_layers + n_layers) % n_layers

        prev_layer = "observation"

        for idx, size in enumerate(hidden_sizes):
            hidden_layer = Dense(
                output_dim=size,
                init=hidden_init,
                activation=hidden_nl
            )
            if idx == action_merge_layer:
                input, inputs = None, [prev_layer, "action"]
            else:
                input, inputs = prev_layer, []
            graph.add_node(
                hidden_layer,
                name=("h%d" % idx),
                input=input,
                inputs=inputs
            )
            # if bn:
            #     bn_layer = BatchNormalization()
            # else:
            #     bn_layer = Layer()
            # graph.add_node(
            #     bn_layer,
            #     name=("bn%d" % idx),
            #     input=("h%d" % idx)
            # )
            prev_layer = ("h%d" % idx)

        if action_merge_layer == n_layers:
            input, inputs = None, [prev_layer, "action"]
        else:
            input, inputs = prev_layer, []

        output_layer = Dense(
            output_dim=1,
            init=output_init,
            activation=output_nl,
        )
        graph.add_node(
            output_layer,
            name="output",
            input=input,
            inputs=inputs,
            create_output=True
        )

        self._graph = graph
        self._output_layer = output_layer
        self._output_nl = output_nl

        ContinuousQFunction.__init__(self, mdp)
        KerasPowered.__init__(self, graph)
        Serializable.__init__(
            self, mdp=mdp, hidden_sizes=hidden_sizes, hidden_nl=hidden_nl,
            hidden_init=hidden_init, action_merge_layer=action_merge_layer,
            output_nl=output_nl, output_init=output_init, bn=bn)

    @property
    @overrides
    def normalizable(self):
        return self._output_nl == 'linear'

    @overrides
    def get_qval_sym(self, obs_var, action_var, train=False):
        self._graph.inputs["observation"].input = obs_var
        self._graph.inputs["action"].input = action_var
        qvals = self._graph.get_output(train=train)["output"]
        return TT.reshape(qvals, (-1,))

    @overrides
    def get_output_W(self):
        return self._output_layer.W.get_value()

    # pylint: disable=no-member
    @overrides
    def get_output_b(self):
        return self._output_layer.b.get_value()

    @overrides
    def set_output_W(self, W_new):
        self._output_layer.W.set_value(W_new)

    @overrides
    def set_output_b(self, b_new):
        self._output_layer.b.set_value(b_new)
    # pylint: enable=no-member

    def get_default_updates(self, obs_var, action_var, train=False):
        self._graph.inputs["observation"].input = obs_var
        self._graph.inputs["action"].input = action_var
        for layer in self._graph.nodes.values():
            layer.build()
        return dict(self._graph.updates)
