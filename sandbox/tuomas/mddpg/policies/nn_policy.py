import tensorflow as tf
import numpy as np

from sandbox.haoran.mddpg.core.neuralnet import NeuralNetwork
from sandbox.haoran.mddpg.core.tf_util import he_uniform_initializer, mlp, linear, weight_variable
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy


class NNPolicy(NeuralNetwork, Policy):
    def __init__(self, scope_name, observation_dim, action_dim, **kwargs):
        Serializable.quick_init(self, locals())
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        with tf.variable_scope(scope_name) as variable_scope:
            super(NNPolicy, self).__init__(
                variable_scope.original_name_scope, **kwargs)
            self.observations_placeholder = tf.placeholder(
                tf.float32,
                shape=[None, self.observation_dim],
                name="actor_obs"
            )
            self._output = self.create_network()
            self.variable_scope = variable_scope

    def get_action(self, observation):
        return self.sess.run(self.output,
                             {self.observations_placeholder: [observation]}), {}

    def get_actions(self, observations):
        return self.sess.run(self.output,
                             {self.observations_placeholder: observations}), {}

    def create_network(self):
        """
        Use self.observations_placeholder to create an output. (To be
        refactored soon so that the input is passed in.)

        :return: TensorFlow tensor.
        """
        raise NotImplementedError

    def plot_samples(self, ax_lst, obs_lst, K):
        for ax, obs in zip(ax_lst, obs_lst):
            obs = obs.reshape((-1, self.observation_dim))
            obs = np.tile(obs, (K, 1))

            actions, _ = self.get_actions(obs)

            x, y = actions[:, 0], actions[:, 1]
            ax.plot(x, y, '*')


class FeedForwardPolicy(NNPolicy):
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_dim,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
    ):
        Serializable.quick_init(self, locals())
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        super(FeedForwardPolicy, self).__init__(scope_name,
                                                observation_dim,
                                                action_dim)

    def create_network(self):
        observation_output = mlp(
            self.observations_placeholder,
            self.observation_dim,
            self.observation_hidden_sizes,
            self.hidden_nonlinearity,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init,
        )
        return self.output_nonlinearity(linear(
            observation_output,
            self.observation_hidden_sizes[-1],
            self.action_dim,
            W_initializer=self.output_W_init,
            b_initializer=self.output_b_init,
        ))

    def get_feed_dict(self, observations):
        return {self.observations_placeholder: observations}


class SumPolicy(NNPolicy):
    """Just output the sum of the inputs. This is used to debug."""

    def create_network(self):
        W_obs = weight_variable((self.observation_dim, 1),
                                initializer=tf.constant_initializer(1.))
        return tf.matmul(self.observations_placeholder, W_obs)
