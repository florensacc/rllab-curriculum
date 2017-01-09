import tensorflow as tf

from sandbox.haoran.mddpg.core.neuralnet import NeuralNetwork
from sandbox.haoran.mddpg.core.tf_util import he_uniform_initializer, mlp, linear, weight_variable
from rllab.core.serializable import Serializable


class NNCritic(NeuralNetwork):
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_dim,
            action_input=None,
            observation_input=None,
            reuse=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super(NNCritic, self).__init__(scope_name, **kwargs)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.action_input = action_input
        self.observation_input = observation_input
        self.reuse = reuse

        with tf.variable_scope(self.scope_name, reuse=reuse) as variable_scope:
            if action_input is None:
                self.actions_placeholder = tf.placeholder(
                    tf.float32,
                    shape=[None, action_dim],
                    name='critic_actions',
                )
            else:
                self.actions_placeholder = action_input
            if observation_input is None:
                self.observations_placeholder = tf.placeholder(
                    tf.float32,
                    shape=[None, observation_dim],
                    name='critic_observations',
                )
            else:
                self.observations_placeholder = observation_input
            self._output = self.create_network(self.actions_placeholder,
                                               self.observations_placeholder)
            self.variable_scope = variable_scope

    def get_weight_tied_copy(self, action_input):
        """
        HT: basically, re-run __init__ with specified kwargs. In particular,
        the variable scope doesn't change, and self.observations_placeholder
        and NN params are reused.
        """
        # This original version may try to pickle tensors, causing bugs
        # return self.get_copy(
        #     scope_name=self.scope_name,
        #     action_input=action_input,
        #     reuse=True)
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            action_input=action_input,
            reuse=True,
        )

    def create_network(self, action_input):
        raise NotImplementedError

class FeedForwardCritic(NNCritic):
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_dim,
            action_input=None,
            reuse=False,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
    ):
        Serializable.quick_init(self, locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        super().__init__(scope_name, observation_dim, action_dim,
                         action_input=action_input, reuse=reuse)

    def get_weight_tied_copy(self, action_input):
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            action_input=action_input,
            reuse=True,
            hidden_W_init=self.hidden_W_init,
            hidden_b_init=self.hidden_b_init,
            output_W_init=self.output_W_init,
            output_b_init=self.output_b_init,
            embedded_hidden_sizes=self.embedded_hidden_sizes,
            observation_hidden_sizes=self.observation_hidden_sizes,
            hidden_nonlinearity=self.hidden_nonlinearity,
        )

    def create_network(self, action_input):
        with tf.variable_scope("observation_mlp") as _:
            observation_output = mlp(self.observations_placeholder,
                                     self.observation_dim,
                                     self.observation_hidden_sizes,
                                     self.hidden_nonlinearity,
                                     W_initializer=self.hidden_W_init,
                                     b_initializer=self.hidden_b_init,
                                     reuse_variables=True)
        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = self.action_dim + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp") as _:
            fused_output = mlp(embedded,
                               embedded_dim,
                               self.embedded_hidden_sizes,
                               self.hidden_nonlinearity,
                               W_initializer=self.hidden_W_init,
                               b_initializer=self.hidden_b_init,
                               reuse_variables=True)

        with tf.variable_scope("output_linear") as _:
            return linear(fused_output,
                          self.embedded_hidden_sizes[-1],
                          1,
                          W_initializer=self.output_W_init,
                          b_initializer=self.output_b_init,
                          reuse_variables=True)


class SumCritic(NNCritic):
    """Just output the sum of the inputs. This is used to debug."""

    def create_network(self, action_input):
        with tf.variable_scope("actions_layer") as _:
            W_actions = weight_variable(
                (self.action_dim, 1),
                initializer=tf.constant_initializer(1.),
                reuse_variables=True)
        with tf.variable_scope("observation_layer") as _:
            W_obs = weight_variable(
                (self.observation_dim, 1),
                initializer=tf.constant_initializer(1.),
                reuse_variables=True)

        return (tf.matmul(action_input, W_actions) +
                tf.matmul(self.observations_placeholder, W_obs))
