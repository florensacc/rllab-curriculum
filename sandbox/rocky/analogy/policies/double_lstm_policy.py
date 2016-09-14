from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.policies.base import Policy
from sandbox.rocky.tf.misc import tensor_utils
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import MLP
import tensorflow as tf
import numpy as np
# from rocky.core.network import MLP


class DoubleLSTMPolicy(Policy):
    """
    Use a separate LSTM for processing the demonstration trajectory, and then another LSTM for execution.
    """

    def __init__(self, env, horizon, lstm_size):
        self.env = env

        self.obs_dim = env.observation_space.flat_dim
        self.action_dim = env.action_space.flat_dim

        self.horizon = horizon
        self.lstm_size = lstm_size
        self.demo_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=True)
        self.analogy_lstm = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, state_is_tuple=True)

        self.demo_obs_var = env.observation_space.new_tensor_variable(name="demo_obs", extra_dims=2)
        self.demo_action_var = env.action_space.new_tensor_variable(name="demo_action", extra_dims=2)
        self.analogy_obs_var = env.observation_space.new_tensor_variable(name="analogy_obs", extra_dims=2)

        # Variables for the initial state of demo LSTM
        self.demo_c0_var = tf.Variable(np.zeros((1, self.lstm_size)), name="demo_c0", dtype=tf.float32)
        self.demo_h0_var = tf.Variable(np.zeros((1, self.lstm_size)), name="demo_h0", dtype=tf.float32)

        # Variables for the initial state of demo LSTM
        self.analogy_c0_var = tf.Variable(np.zeros((1, self.lstm_size)), name="analogy_c0", dtype=tf.float32)
        self.analogy_h0_var = tf.Variable(np.zeros((1, self.lstm_size)), name="analogy_h0", dtype=tf.float32)

        # Placeholders for the current state of LSTM, to be used when generating actions using the policy
        self.demo_ct_var = tf.Variable(np.zeros((1, self.lstm_size)), name="demo_ct", dtype=tf.float32)
        self.demo_ht_var = tf.Variable(np.zeros((1, self.lstm_size)), name="demo_ht", dtype=tf.float32)

        # Placeholders for the current state of LSTM, to be used when generating actions using the policy
        self.analogy_ct_var = tf.Variable(np.zeros((1, self.lstm_size)), name="analogy_ct", dtype=tf.float32)
        self.analogy_ht_var = tf.Variable(np.zeros((1, self.lstm_size)), name="analogy_ht", dtype=tf.float32)
        self.demo_output_var = tf.Variable(np.zeros((1, self.lstm_size)), name="demo_output", dtype=tf.float32)

        self.demo_scope = "DemoLSTM"
        self.analogy_scope = "AnalogyLSTM"

        self.action_network = MLP(
            name="action_network",
            input_shape=(self.lstm_size,),
            hidden_sizes=(),
            hidden_nonlinearity=None,
            output_nonlinearity=None,
            output_dim=self.action_dim,
        )

        # self.action_sym()

        # initialize variables
        self.demo_lstm(
            tf.concat(1, [self.demo_obs_var[:, 0, :], self.demo_action_var[:, 0, :]]),
            (self.demo_c0_var, self.demo_h0_var),
            scope=self.demo_scope
        )
        self.demo_lstm_variables = [v for v in tf.all_variables() if v.name.startswith(self.demo_scope)]

        self.analogy_lstm(
            tf.concat(1, [self.analogy_obs_var[:, 0, :], self.demo_output_var]),
            (self.analogy_c0_var, self.analogy_h0_var),
            scope=self.analogy_scope
        )
        self.analogy_lstm_variables = [v for v in tf.all_variables() if v.name.startswith(self.analogy_scope)]

        self.demo_applied = False

        self.action_var = self.action_sym(
            self.analogy_obs_var,
            init_state=(self.analogy_ct_var, self.analogy_ht_var),
            update_state=True,
            horizon=1,
            demo_output_var=self.demo_output_var,
        )

        self.apply_demo_op = tf.assign(
            self.demo_output_var,
            self.demo_output_sym(self.demo_obs_var, self.demo_action_var)
        )
        Policy.__init__(self, self.env.spec)

    def apply_demo(self, demo_obs, demo_actions):
        sess = tf.get_default_session()
        sess.run(
            self.apply_demo_op,
            feed_dict={self.demo_obs_var: [demo_obs], self.demo_action_var: [demo_actions]}
        )
        self.demo_applied = True

    def get_params_internal(self, **tags):
        params = list(self.action_network.get_params_internal(**tags))
        # Only include the parameters if trainable=True or not set
        if tags.get('trainable', True):
            params.extend(self.demo_lstm_variables)
            params.extend(self.analogy_lstm_variables)
        # Only include the parameters if trainable=False or not set
        if not tags.get('trainable', False):
            params.extend(self.demo_c0_var)
            params.extend(self.demo_h0_var)
            params.extend(self.analogy_c0_var)
            params.extend(self.analogy_h0_var)
            params.extend(self.demo_ct_var)
            params.extend(self.demo_ht_var)
            params.extend(self.analogy_ct_var)
            params.extend(self.analogy_ht_var)
            params.extend(self.demo_output_var)
        return params

    def reset(self, dones=None):
        asser
        sess = tf.get_default_session()
        sess.run([
            tf.assign(self.demo_ct_var, self.demo_c0_var),
            tf.assign(self.demo_ht_var, self.demo_h0_var),
            tf.assign(self.analogy_ct_var, self.analogy_c0_var),
            tf.assign(self.analogy_ht_var, self.analogy_h0_var),
            tf.assign(self.demo_output_var, np.zeros((1, self.lstm_size)))
        ])
        self.demo_applied = False

    def get_action(self, obs):
        assert self.demo_applied
        flat_obs = self.env.observation_space.flatten(obs)
        sess = tf.get_default_session()
        action = sess.run(
            self.action_var,
            feed_dict={self.analogy_obs_var: [[flat_obs]]}
        )[0, 0]
        return self.env.action_space.flatten(action)

    def demo_output_sym(self, demo_obs_var, demo_action_var):
        N = tf.shape(demo_obs_var)[0]
        demo_input_var = tf.concat(2, [demo_obs_var, demo_action_var])
        demo_c0_multi = tf.tile(self.demo_c0_var, tf.pack([N, 1]))
        demo_h0_multi = tf.tile(self.demo_h0_var, tf.pack([N, 1]))
        demo_c0_multi.set_shape((None, self.lstm_size))
        demo_h0_multi.set_shape((None, self.lstm_size))
        demo_state = (demo_c0_multi, demo_h0_multi)
        for i in range(self.horizon):
            demo_output, demo_state = self.demo_lstm(
                demo_input_var[:, i, :],
                demo_state,
                scope=self.demo_scope
            )
        return demo_output

    def action_sym(
            self,
            analogy_obs_var,
            horizon=None,
            demo_obs_var=None,
            demo_action_var=None,
            demo_output_var=None,
            init_state=None,
            update_state=False,
    ):
        # First, process demo obs / actions to get final state
        N = tf.shape(analogy_obs_var)[0]
        tf.get_variable_scope().reuse_variables()

        if demo_output_var is None:
            assert demo_obs_var is not None and demo_action_var is not None, \
                "Must provide either demo_output or both demo_obs_var and demo_action_var!"
            demo_output_var = self.demo_output_sym(demo_obs_var, demo_action_var)

        if horizon is None:
            horizon = self.horizon

        # Now, demo_output should be of size batch_size x lstm_size

        # Process the analogy obs and produce actions
        if init_state is None:
            analogy_c0_multi = tf.tile(self.analogy_c0_var, tf.pack([N, 1]))
            analogy_h0_multi = tf.tile(self.analogy_h0_var, tf.pack([N, 1]))
            analogy_c0_multi.set_shape((None, self.lstm_size))
            analogy_h0_multi.set_shape((None, self.lstm_size))
            analogy_state = (analogy_c0_multi, analogy_h0_multi)
        else:
            analogy_state = init_state

        analogy_outputs = []
        for i in range(horizon):
            # Pass demo_output every time step
            analogy_output, analogy_state = self.analogy_lstm(
                tf.concat(1, [analogy_obs_var[:, i, :], demo_output_var]),
                analogy_state,
                scope=self.analogy_scope
            )
            analogy_outputs.append(tf.expand_dims(analogy_output, 1))

        # analogy_outputs: batch_size x #timesteps x lstm_size
        analogy_outputs = tf.concat(1, analogy_outputs)

        update_ops = []
        if update_state:
            assert init_state is not None
            update_ops.extend([tf.assign(x, y) for x, y in zip(init_state, analogy_state)])

        with tf.control_dependencies(update_ops):
            # action_var: batch_size x #timesteps x action_dim
            action_var = tf.reshape(
                L.get_output(
                    self.action_network.output_layer, tf.reshape(analogy_outputs, (-1, self.lstm_size))
                ),
                tf.pack([N, -1, self.action_dim])
            )
            return action_var
