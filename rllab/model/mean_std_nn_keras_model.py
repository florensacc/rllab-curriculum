from keras.models import Graph
from rllab.model.base import Model
from rllab.core.keras_powered import KerasPowered
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class MeanStdNNKerasModel(Model, KerasPowered, Serializable):

    def __init__(self, mdp):

        graph = Graph()
        graph.add_input("observation", input_shape=mdp.observation_shape)
        graph.add_input("action", input_shape=(mdp.action_dim,))

        Model.__init__(self, mdp)
        KerasPowered.__init__(self, graph)
        Serializable.__init__(self, mdp=mdp)
