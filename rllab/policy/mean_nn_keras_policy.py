import itertools
from keras.models import Graph
from keras.layers.core import Dense
from keras.layers.normalization import BatchNormalization
from rllab.policy.base import DeterministicPolicy
from rllab.core.keras_powered import KerasPowered
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.ext import compile_function


class MeanNNKerasPolicy(DeterministicPolicy, KerasPowered, Serializable):
    """
    A policy that just outputs a mean (i.e. a deterministic policy)
    """

    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('hidden_nl', type=str, nargs='*',
                  help='list of nonlinearities for the hidden layers')
    @autoargs.arg('hidden_init', type=str, nargs='*',
                  help='list of initializers for the hidden layers weights')
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
            hidden_nl=['relu'],
            hidden_init=['he_uniform'],
            output_nl='linear',
            output_init='he_uniform',
            bn=False):
        # pylint: enable=dangerous-default-value

        assert len(mdp.observation_shape) == 1

        graph = Graph()

        graph.add_input("observation", input_shape=mdp.observation_shape)

        if len(hidden_nl) == 1:
            hidden_nl *= len(hidden_sizes)
        assert len(hidden_nl) == len(hidden_sizes)

        if len(hidden_init) == 1:
            hidden_init *= len(hidden_sizes)
        assert len(hidden_init) == len(hidden_sizes)

        prev_layer = "observation"

        for idx, size, nl, init in zip(
                itertools.count(), hidden_sizes, hidden_nl,
                hidden_init):
            if idx == 0:
                input_dim = mdp.observation_shape[0]
            else:
                input_dim = None
            hidden_layer = Dense(
                output_dim=size,
                init=init,
                activation=nl,
                input_dim=input_dim,
            )
            graph.add_node(hidden_layer, name=("h%d" % idx), input=prev_layer)
            if bn:
                bn_layer = BatchNormalization()
                graph.add_node(
                    bn_layer,
                    name=("bn%d" % idx),
                    input=("h%d" % idx)
                )
                prev_layer = ("bn%d" % idx)
            else:
                prev_layer = ("h%d" % idx)

        output_layer = Dense(
            output_dim=mdp.action_dim,
            init=output_init,
            activation=output_nl,
        )

        graph.add_node(
            output_layer,
            name="output",
            input=prev_layer,
            create_output=True
        )

        input_var = graph.inputs["observation"].get_input(train=False)
        action_var = graph.get_output(train=False)
        self._f_actions = compile_function([input_var], action_var)

        self._graph = graph

        super(MeanNNKerasPolicy, self).__init__(mdp)
        KerasPowered.__init__(self, graph)
        Serializable.__init__(
            self, mdp=mdp, hidden_sizes=hidden_sizes, hidden_nl=hidden_nl,
            hidden_init=hidden_init, output_nl=output_nl,
            output_init=output_init, bn=bn)

    @property
    @overrides
    def action_dim(self):
        return self._action_dim

    @property
    @overrides
    def action_dtype(self):
        return self._action_dtype

    @overrides
    def get_action_sym(self, input_var, train=False):
        self._graph.inputs["observation"].input = input_var
        return self._graph.get_output(train=train)

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

    def get_default_updates(self, obs_var, train=False):
        self._graph.inputs["observation"].input = obs_var
        for layer in self._graph.nodes.values():
            layer.build()
        return dict(self._graph.updates)
