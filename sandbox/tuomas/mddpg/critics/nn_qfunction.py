import tensorflow as tf

from sandbox.haoran.mddpg.core.neuralnet import NeuralNetwork
from sandbox.haoran.mddpg.core.tf_util import he_uniform_initializer, mlp, linear, weight_variable
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides

import numpy as np


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
            self._output = self.create_network(
                self.actions_placeholder,
                self.observations_placeholder,
            )
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

    def create_network(self, action_input, observation_input):
        raise NotImplementedError

    def get_feed_dict(self, obs, action=None):
        feed = {self.observations_placeholder: obs}
        if action is not None:
            feed[self.actions_placeholder] = action

        return feed


class FeedForwardCritic(NNCritic):
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_dim,
            action_input=None,
            observation_input=None,
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
                         action_input=action_input,
                         observation_input=observation_input,
                         reuse=reuse)

    def get_weight_tied_copy(self, action_input, observation_input):
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            action_input=action_input,
            observation_input=observation_input,
            reuse=True,
            hidden_W_init=self.hidden_W_init,
            hidden_b_init=self.hidden_b_init,
            output_W_init=self.output_W_init,
            output_b_init=self.output_b_init,
            embedded_hidden_sizes=self.embedded_hidden_sizes,
            observation_hidden_sizes=self.observation_hidden_sizes,
            hidden_nonlinearity=self.hidden_nonlinearity,
        )

    def create_network(self, action_input, observation_input):
        with tf.variable_scope("observation_mlp") as _:
            observation_output = mlp(observation_input,
                                     self.observation_dim,
                                     self.observation_hidden_sizes,
                                     self.hidden_nonlinearity,
                                     W_initializer=self.hidden_W_init,
                                     b_initializer=self.hidden_b_init,
                                     reuse_variables=True)
        embedded = tf.concat(1, [observation_output, action_input])
        if len(self.observation_hidden_sizes) == 0:
            embedded_dim = self.action_dim + self.observation_dim
        else:
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

    def plot(self, ax_lst, obs_lst, action_dims, xlim, ylim):
        """
        Plots level curves of critic output.

        :param ax_lst: List of plt Axes instances.
        :param obs_lst: List of input observations at which the critic is
            evaluated.
        :return:
        """
        assert len(action_dims) == 2

        xx = np.arange(xlim[0], xlim[1], 0.05)
        yy = np.arange(ylim[0], ylim[1], 0.05)
        X, Y = np.meshgrid(xx, yy)

        actions = np.zeros((X.size, self.action_dim))
        actions[:, action_dims[0]] = X.ravel()
        actions[:, action_dims[1]] = Y.ravel()

        feed = {self.actions_placeholder: actions}

        for ax, obs in zip(ax_lst, obs_lst):
            obs = obs.reshape((-1, self.observation_dim))
            obs = np.tile(obs, (actions.shape[0], 1))

            feed[self.observations_placeholder] = obs
            Q = self.sess.run(self.output, feed).reshape(X.shape)

            cs = ax.contour(X, Y, Q, 20)
            ax.clabel(cs, inline=1, fontsize=10, fmt='%.2f')


class MultiCritic(NNCritic):
    """
    Train multiple independent critics simultaneously. Requires that the
    reward dimensions matches the number critics. Output is simply the sum
    of the critics' outputs.

    TODO: also allows varying temperature
    TODO: also works with a single critic
    """
    def __init__(self, critics, default_temperatures=None):
        """

        :param critics: List of critics.
        """
        Serializable.quick_init(self, locals())

        self._critics = critics
        self._M = len(critics)
        self._create_temperature_placeholder()

        if default_temperatures is None:
            default_temperatures = np.ones(self._M)
        self._default_temperatures = default_temperatures

        # Make sure that all critic outputs have two axes.
        outputs_list = []
        for c in self._critics:
            if len(c.output.get_shape().dims) == 1:
                outputs_list.append(tf.reshape(c.output, (-1, 1)))
            else:
                outputs_list.append(c.output)

        self._outputs = tf.concat(1, outputs_list)
        # Notice that the temperature is actually 1 / T.
        self._output = tf.reduce_sum(self._outputs * self._temp_pl, axis=1)

    def _create_temperature_placeholder(self):
        with tf.variable_scope(self.scope_name, reuse=False):
            self._temp_pl = tf.placeholder(
                tf.float32,
                shape=(None, self._M),
                name='critic_temps'
            )

    def get_weight_tied_copy(self, action_input, observation_input):
        return MultiCritic([
            c.get_weight_tied_copy(action_input, observation_input)
            for c in self._critics
        ])

    @overrides
    def get_params_internal(self, **tags):
        all_params = []
        for c in self._critics:
            all_params += c.get_params_internal()
        return all_params

    @overrides
    def get_param_values(self, **tags):
        all_params = []
        for c in self._critics:
            all_params.append(c.get_param_values(**tags))
        return all_params

    @overrides
    def set_param_values(self, params, **tags):
        for c, p in zip(self._critics, params):
            c.set_param_values(p, **tags)

    @overrides
    def get_copy(self, scope_name, **kwargs):
        c_cpy = [c.get_copy(scope_name=scope_name + c.scope_name, **kwargs)
                 for c in self._critics]

        return MultiCritic(c_cpy, self._default_temperatures.copy())

    @overrides
    def get_feed_dict(self, obs, action=None, temp=None):
        if temp is None:
            temp = self._default_temperatures

        # Make sure the dimension is right.
        temp = temp.reshape((-1, self._M))

        feed = {self._temp_pl: temp}
        for c in self._critics:
            feed.update(c.get_feed_dict(obs, action))

        return feed

    @property
    def scope_name(self):
        return ''

    @property
    def outputs(self):
        return self._outputs  # N x M

    @property
    def dim(self):
        return self._M


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
