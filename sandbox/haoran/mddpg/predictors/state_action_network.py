import abc
import tensorflow as tf
from core.neuralnet import NeuralNetwork
from rllab.core.serializable import Serializable


class StateActionNetwork(NeuralNetwork):
    """
    A map from (state, action) to a vector
    """

    def __init__(
            self,
            scope_name,
            env_spec,
            output_dim,
            action_input=None,
            observation_input=None,
            reuse=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.observation_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.output_dim = output_dim
        self.reuse = reuse
        self.action_input = action_input
        self.observation_input = observation_input
        with tf.name_scope(scope_name):
            if self.action_input is None:
                self.action_input = tf.placeholder(
                    tf.float32,
                    [None, self.action_dim],
                    "_actions")
            if self.observation_input is None:
                self.observation_input = tf.placeholder(
                    tf.float32,
                    [None, self.observation_dim],
                    "_observation")

        with tf.variable_scope(scope_name, reuse=reuse) as variable_scope:
            super(StateActionNetwork, self).__init__(
                variable_scope.original_name_scope, **kwargs)
            self._output = self._create_network()
            self.variable_scope = variable_scope

    @property
    def output(self):
        return self._output

    @abc.abstractmethod
    def _create_network(self):
        return
