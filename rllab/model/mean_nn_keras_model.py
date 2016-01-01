import theano.tensor as TT
from keras.models import Graph
from keras.layers.core import Dense
from rllab.model.base import Model
from rllab.core.keras_powered import KerasPowered
from rllab.core.serializable import Serializable


class MeanNNKerasModel(Model, KerasPowered, Serializable):

    """
    A deterministic model, parameterized by a neural network.
    """

    def __init__(self, mdp):

        assert len(mdp.observation_shape) == 1

        graph = Graph()
        graph.add_input("observation", input_shape=mdp.observation_shape)
        graph.add_input("action", input_shape=(mdp.action_dim,))

        graph.add_node(Dense(
            output_dim=100,
            init='he_uniform',
            activation='relu',
        ), name="h0", input="observation")

        graph.add_node(Dense(
            output_dim=100,
            init='he_uniform',
            activation='relu',
        ), name="h1", inputs=["h0", "action"])

        graph.add_node(Dense(
            output_dim=mdp.observation_shape[0],
            init='he_uniform',
            activation='tanh',
        ), name="output_obs", input="h1", create_output=True)
        graph.add_node(Dense(
            output_dim=1,
            init='he_uniform',
            activation='tanh',
        ), name="output_reward", input="h1", create_output=True)

        self._graph = graph

        Model.__init__(self, mdp)
        KerasPowered.__init__(self, graph)
        Serializable.__init__(self, mdp=mdp)

    def _set_inputs(self, obs_var, action_var):
        self._graph.inputs["observation"].input = obs_var
        self._graph.inputs["action"].input = action_var
        for layer in self._graph.nodes.values():
            layer.build()

    def predict_obs_sym(self, obs_var, action_var, train=False):
        self._set_inputs(obs_var, action_var)
        return self._graph.get_output(train=train)["output_obs"]

    def predict_reward_sym(self, obs_var, action_var, train=False):
        self._set_inputs(obs_var, action_var)
        rewards = self._graph.get_output(train=train)["output_reward"]
        return TT.reshape(rewards, (-1,))

    def obs_regression_obj(self, obs_var, action_var, next_obs_var,
                           train=False):
        predicted = self.predict_obs_sym(obs_var, action_var, train=train)
        return TT.mean(TT.square(predicted - next_obs_var))

    def reward_regression_obj(self, obs_var, action_var, reward_var,
                              train=False):
        predicted = self.predict_reward_sym(obs_var, action_var, train=train)
        return TT.mean(TT.square(predicted - reward_var))
