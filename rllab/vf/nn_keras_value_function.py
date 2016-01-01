import theano.tensor as TT
from keras.models import Graph
from keras.layers.core import Dense
from rllab.vf.base import ValueFunction
from rllab.core.keras_powered import KerasPowered
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class NNKerasValueFunction(ValueFunction, KerasPowered, Serializable):

    def __init__(self, mdp):

        graph = Graph()
        graph.add_input("observation", input_shape=mdp.observation_shape)
        graph.add_node(Dense(
            output_dim=200,
            init='he_uniform',
            activation='relu',
        ), name="h0", input="observation")
        graph.add_node(Dense(
            output_dim=100,
            init='he_uniform',
            activation='relu',
        ), name="h1", input="h0")
        graph.add_node(Dense(
            output_dim=1,
            init='he_uniform',
            activation='linear',
        ), name="output", input="h1", create_output=True)

        self._graph = graph

        ValueFunction.__init__(self, mdp)
        KerasPowered.__init__(self, graph)
        Serializable.__init__(self, mdp)

    def _set_inputs(self, obs_var):
        self._graph.inputs["observation"].input = obs_var
        for layer in self._graph.nodes.values():
            layer.build()

    @overrides
    def get_val_sym(self, obs_var, train=False):
        self._set_inputs(obs_var)
        vals = self._graph.get_output(train=train)["output"]
        return TT.reshape(vals, (-1,))
