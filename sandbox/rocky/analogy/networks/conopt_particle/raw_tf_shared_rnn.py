import tensorflow as tf
import numpy as np
import prettytensor as pt
from sandbox.rocky.tf.core.parameterized import Parameterized


class SummaryNetwork(Parameterized):
    def __init__(self, env_spec, state_dim=100, n_steps=30):
        super().__init__()
        # with tf.variable_scope("summary_network"):
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        obs_var = tf.placeholder(dtype=tf.float32, shape=(None, None, obs_dim), name="obs")
        action_var = tf.placeholder(dtype=tf.float32, shape=(None, None, action_dim), name="action")

        summary_var = tf.Variable(
            initial_value=np.zeros((0, state_dim), dtype=np.float32),
            validate_shape=False,
            name="summary",
            trainable=False
        )
        self.summary_var = summary_var
        self.state_dim = state_dim
        self.n_steps = n_steps

        self.obs_var = obs_var
        self.action_var = action_var
        self.input_vars = [obs_var, action_var]
        self.rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.state_dim)
        self.output_dim = state_dim
        # bootstrap
        # self.vs = tf.variable_scope("shared_rnn")
        self.get_update_op(obs_var, action_var, reuse=False)
        #     pass

    def get_update_op(self, obs_var=None, action_var=None, reuse=True, **kwargs):
        summary_var = self.get_output(obs_var=obs_var, action_var=action_var, reuse=reuse)
        return tf.assign(self.summary_var, summary_var, validate_shape=False)

    def get_output(self, obs_var=None, action_var=None, reuse=True, **kwargs):
        # if reuse:
        #     tf.get_variable_scope().reuse_variables()
        # else:
        #     tf.get_variable_scope()._reuse = False
        if obs_var is None:
            obs_var = self.obs_var
        if action_var is None:
            action_var = self.action_var
        # with tf.variable_scope("shared_rnn", reuse=reuse) as vs:
        inp = tf.concat(2, [obs_var, action_var])

        N = tf.shape(obs_var)[0]

        inits = tf.zeros(tf.pack([N, self.state_dim]))

        with tf.variable_scope("shared_rnn", reuse=reuse) as vs:
            # bootstrap the weights
            self.rnn_cell(inp[:, 0, :], inits, scope=vs)

        for idx in range(self.n_steps):
            with tf.variable_scope("shared_rnn", reuse=reuse) as vs:
                # bootstrap the weights
                self.rnn_cell(inp[:, 0, :], inits, scope=vs)

            # vs.reuse_variables()

            def step(hprev, x):
                vs.reuse_variables()
                return self.rnn_cell(x, hprev, scope=vs)[1]

        hs = tf.scan(
            step,
            elems=tf.transpose(inp, (1, 0, 2)),
            initializer=inits,
        )
        return tf.reverse(hs, [True, False, False])[0, :, :]

    def get_params_internal(self, **tags):
        all_vars = [v for v in tf.all_variables() if v.name.startswith("shared_rnn")]
        trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith("shared_rnn")]
        if tags.get('trainable') is True:
            vars = trainable_vars#[v for v in traif v.trainable == tags['trainable']]
        elif tags.get('trainable') is False:
            vars = list(set(all_vars) - set(trainable_vars))
        else:
            vars = all_vars
        # if not tags.get('trainable', False):
        #     vars.append(self.summary_var)
        return sorted(vars, key=lambda x: x.name)


class ActionNetwork(Parameterized):
    def __init__(self, env_spec, summary_network):
        super().__init__()
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        self.prev_action_var = tf.Variable(
            initial_value=np.zeros((0, action_dim), dtype=np.float32),
            validate_shape=False,
            name="prev_action",
            trainable=False
        )
        self.prev_state_var = tf.Variable(
            initial_value=np.zeros((0, summary_network.state_dim), dtype=np.float32),
            validate_shape=False,
            name="prev_state",
            trainable=False
        )
        self.state_dim = summary_network.state_dim
        self.summary_network = summary_network
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        obs_var = env_spec.observation_space.new_tensor_variable("obs", extra_dims=1)
        self.get_step_op(obs_var=obs_var, reuse=False)

    def get_partial_reset_op(self, dones_var):
        # upon reset: set corresponding entry to zero
        N = tf.shape(dones_var)[0]
        dones_var = tf.expand_dims(dones_var, 1)
        initial_prev_action = tf.zeros(tf.pack([N, self.action_dim]))
        initial_prev_state = self.summary_network.summary_var

        return tf.group(
            tf.assign(
                self.prev_action_var,
                self.prev_action_var * (1. - dones_var) + initial_prev_action * dones_var,
                validate_shape=False
            ),
            tf.assign(
                self.prev_state_var,
                self.prev_state_var * (1. - dones_var) + initial_prev_state * dones_var,
                validate_shape=False
            )
        )

    def get_full_reset_op(self, dones_var):
        N = tf.shape(dones_var)[0]
        initial_prev_action = tf.zeros(tf.pack([N, self.action_dim]))
        initial_prev_state = self.summary_network.summary_var

        return tf.group(
            tf.assign(
                self.prev_action_var,
                initial_prev_action,
                validate_shape=False
            ),
            tf.assign(
                self.prev_state_var,
                initial_prev_state,
                validate_shape=False
            )
        )

    def get_step_op(self, obs_var, reuse=True, **kwargs):
        #     rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.summary_network.state_dim)

        inp = tf.concat(1, [obs_var, self.prev_action_var])
        inp.set_shape((None, self.obs_dim + self.action_dim))

        # with tf.variable_scope("shared_rnn", reuse=True) as vs:
        prev_state = tf.convert_to_tensor(self.prev_state_var)
        prev_state.set_shape((None, self.state_dim))

        with tf.variable_scope("shared_rnn", reuse=True) as vs:
            new_state = self.summary_network.rnn_cell(inp, prev_state, scope=vs)[1]

        # import ipdb; ipdb.set_trace()

        with tf.variable_scope("action_net", reuse=reuse):
            action_var = pt.wrap(new_state).fully_connected(self.action_dim)

        with tf.control_dependencies([
            tf.assign(self.prev_action_var, action_var),
            tf.assign(self.prev_state_var, new_state)
        ]):
            action_var = tf.identity(action_var)
        return action_var, self.prev_action_var

    @property
    def state_info_specs(self):
        return [
            ("prev_action", self.action_dim),
        ]

    @property
    def recurrent(self):
        return True

    @property
    def state_info_keys(self):
        return [k for (k, _) in self.state_info_specs]

    def get_output(self, obs_var, summary_var, state_info_vars, reuse=True, **kwargs):
        inp = tf.concat(2, [obs_var, state_info_vars["prev_action"]])
        inp.set_shape((None, None, self.obs_dim + self.action_dim))
        N = tf.shape(obs_var)[0]

        # with tf.variable_scope("shared_rnn", reuse=True) as vs:
        #     rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=self.summary_network.state_dim)

        inits = tf.zeros(tf.pack([N, self.summary_network.state_dim]))

        # with tf.variable_scope("shared_rnn", reuse=True) as vs:
        with tf.variable_scope("shared_rnn", reuse=True) as vs:
            def step(hprev, x):
                return self.summary_network.rnn_cell(x, hprev, scope=vs)[1]

        hs = tf.scan(
            step,
            elems=tf.transpose(inp, (1, 0, 2)),
            initializer=inits,
        )

        hs = tf.transpose(hs, (1, 0, 2))

        with tf.variable_scope("action_net", reuse=reuse):
            flat_hs = tf.reshape(hs, (-1, self.summary_network.state_dim))
            action_var = pt.wrap(flat_hs).fully_connected(self.action_dim)
            action_var = tf.reshape(action_var, tf.pack([tf.shape(obs_var)[0], tf.shape(obs_var)[1], self.action_dim]))

        return action_var#, self.prev_action_var#dict(prev_action=self.prev_action_var)

    def get_params_internal(self, **tags):
        all_vars = [v for v in tf.all_variables() if v.name.startswith("shared_rnn") or v.name.startswith("action_net")]
        trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith("shared_rnn") or
                          v.name.startswith("action_net")]
        if tags.get('trainable') is True:
            vars = trainable_vars
        elif tags.get('trainable') is False:
            vars = list(set(all_vars) - set(trainable_vars))
        else:
            vars = all_vars
        # if not tags.get('trainable', False):
        #     vars.append(self.prev_action_var)
        #     vars.append(self.prev_state_var)
        return sorted(vars, key=lambda x: x.name)


class Net(object):
    def new_networks(self, env_spec):
        summary_network = SummaryNetwork(env_spec=env_spec)
        action_network = ActionNetwork(env_spec=env_spec, summary_network=summary_network)
        return summary_network, action_network
