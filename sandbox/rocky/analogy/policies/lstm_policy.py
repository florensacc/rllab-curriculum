from __future__ import print_function
from __future__ import absolute_import
from rocky.core.parameterized import Parameterized
from rocky.core.gym_ext import unflatten, new_tensor_variable, flat_dim, flatten
import tensorflow as tf
import numpy as np
from rocky.core.network import MLP
import rocky.core.layers as L


class LSTMPolicy(Parameterized):
    def __init__(self, env, horizon, lstm_size):
        self.env = env

        self.obs_dim = flat_dim(env.observation_space)
        self.action_dim = flat_dim(env.action_space)

        self.horizon = horizon
        self.lstm_size = lstm_size
        self.lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=True)

        self.obs_var = new_tensor_variable(env.observation_space, name="obs", extra_dims=2)

        # Variables for the initial state of LSTM
        self.c0_var = tf.Variable(np.zeros((1, self.lstm_size)), name="c0", dtype=tf.float32)
        self.h0_var = tf.Variable(np.zeros((1, self.lstm_size)), name="h0", dtype=tf.float32)

        # Placeholders for the current state of LSTM, to be used when generating actions using the policy
        self.ct_var = tf.Variable(np.zeros((1, self.lstm_size)), name="ct", dtype=tf.float32)
        self.ht_var = tf.Variable(np.zeros((1, self.lstm_size)), name="ht", dtype=tf.float32)

        self.scope = "LSTM"

        self.action_network = MLP(
            name="action_network",
            input_shape=(self.lstm_size,),
            hidden_sizes=(),
            hidden_nonlinearity=None,
            output_nonlinearity=None,
            output_dim=self.action_dim,
        )

        # initialize variables
        self.lstm(self.obs_var[:, 0, :], (self.c0_var, self.h0_var), scope=self.scope)
        self.lstm_variables = [v for v in tf.all_variables() if v.name.startswith(self.scope)]
        self.action_var = self.action_sym(
            self.obs_var,
            init_state=(self.ct_var, self.ht_var),
            horizon=1,
            update_state=True
        )
        Parameterized.__init__(self)

    def get_params_internal(self, **tags):
        params = list(self.action_network.get_params_internal(**tags))
        # Only include the parameters if trainable=True or not set
        if tags.get('trainable', True):
            params.extend(self.lstm_variables)
        # Only include the parameters if trainable=False or not set
        if not tags.get('trainable', False):
            params.extend(self.c0_var)
            params.extend(self.h0_var)
            params.extend(self.ct_var)
            params.extend(self.ht_var)
        return params

    def reset(self):
        sess = tf.get_default_session()
        sess.run([
            tf.assign(self.ct_var, self.c0_var),
            tf.assign(self.ht_var, self.h0_var),
        ])

    def get_action(self, obs):
        flat_obs = flatten(self.env.observation_space, obs)
        sess = tf.get_default_session()
        action = sess.run(
            self.action_var,
            feed_dict={self.obs_var: [[flat_obs]]}
        )[0, 0]
        return action

    def action_sym(self, obs_var, init_state=None, horizon=None, update_state=False):
        outputs = []
        tf.get_variable_scope().reuse_variables()
        N = tf.shape(obs_var)[0]
        if init_state is None:
            c0_multi = tf.tile(self.c0_var, tf.pack([N, 1]))
            h0_multi = tf.tile(self.h0_var, tf.pack([N, 1]))
            c0_multi.set_shape((None, self.lstm_size))
            h0_multi.set_shape((None, self.lstm_size))
            state = (c0_multi, h0_multi)
        else:
            state = init_state
        if horizon is None:
            horizon = self.horizon
        for i in xrange(horizon):
            output, state = self.lstm(obs_var[:, i, :], state, scope=self.scope)
            outputs.append(tf.expand_dims(output, 1))
        outputs = tf.concat(1, outputs)

        update_ops = []
        if update_state:
            assert init_state is not None
            update_ops.extend([tf.assign(x, y) for x, y in zip(init_state, state)])

        with tf.control_dependencies(update_ops):
            action_var = tf.reshape(
                L.get_output(
                    self.action_network.output_layer, tf.reshape(outputs, (-1, self.lstm_size))
                ),
                tf.pack([tf.shape(obs_var)[0], tf.shape(obs_var)[1], self.action_dim])
            )
            return action_var
