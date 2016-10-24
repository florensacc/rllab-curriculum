import sandbox.rocky.tf.core.layers as L
import sandbox.rocky.analogy.core.layers as LL
from sandbox.rocky.analogy.rnn_cells import GRUCell
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.misc import tensor_utils


class ObsEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        obs_dim = env_spec.observation_space.flat_dim

        l_obs_input = L.InputLayer(
            (None, None, obs_dim),
            name="obs_input"
        )

        l_embedding = LL.TemporalDenseLayer(
            l_obs_input,
            num_units=100,
            nonlinearity=tf.nn.relu,
        )

        self.input_vars = [l_obs_input.input_var]
        self.l_obs_input = l_obs_input
        self.embedding_dim = l_embedding.output_shape[-1]
        self.l_embedding = l_embedding
        LayersPowered.__init__(self, [l_embedding])

    def get_output(self, obs_var=None, **kwargs):
        if obs_var is None:
            obs_var = self.l_obs_input
        return L.get_output(
            self.l_embedding,
            {self.l_obs_input: obs_var},
            **kwargs
        )


class ActionNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, obs_embedding_network, cell):
        Serializable.quick_init(self, locals())
        action_dim = env_spec.action_space.flat_dim

        l_embedding = obs_embedding_network.l_embedding

        l_hidden = LL.TfRNNLayer(
            l_embedding,
            cell=cell,
        )

        l_flat_hidden = L.TemporalFlattenLayer(l_hidden)

        l_action_hid = L.DenseLayer(
            l_flat_hidden,
            num_units=100,
            nonlinearity=tf.nn.relu,
        )

        l_flat_action = L.DenseLayer(
            l_action_hid,
            num_units=action_dim,
            nonlinearity=tf.nn.softmax,
        )

        l_action = L.TemporalUnflattenLayer(l_flat_action, obs_embedding_network.l_obs_input)

        self.l_obs_input = obs_embedding_network.l_obs_input

        self.prev_state_var = tf.Variable(
            initial_value=np.zeros((0, cell.state_size), dtype=np.float32),
            validate_shape=False,
            name="prev_state",
            trainable=False
        )
        self.l_embedding = l_embedding
        self.embedding_dim = obs_embedding_network.embedding_dim
        self.l_hidden = l_hidden
        self.l_action = l_action
        self.l_flat_hidden = l_flat_hidden
        self.l_flat_action = l_flat_action
        self.output_layer = l_action
        self.state_dim = cell.state_size
        self.action_dim = action_dim
        self.recurrent_layer = l_hidden
        LayersPowered.__init__(self, [l_action])

    def get_partial_reset_op(self, dones_var):
        # upon reset: set corresponding entry to zero
        N = tf.shape(dones_var)[0]
        dones_var = tf.expand_dims(dones_var, 1)
        initial_prev_state = tf.zeros(tf.pack([N, self.state_dim]))

        return tf.group(
            tf.assign(
                self.prev_state_var,
                self.prev_state_var * (1. - dones_var) + initial_prev_state * dones_var,
                validate_shape=False
            )
        )

    def get_full_reset_op(self, dones_var):
        N = tf.shape(dones_var)[0]
        initial_prev_state = tf.zeros(tf.pack([N, self.state_dim]))

        return tf.group(
            tf.assign(
                self.prev_state_var,
                initial_prev_state,
                validate_shape=False
            )
        )

    def get_step_op(self, obs_var, **kwargs):
        flat_embedding_var = tensor_utils.temporal_flatten_sym(
            L.get_output(
                self.l_embedding,
                {
                    self.l_obs_input: tf.expand_dims(obs_var, 1),
                }, **kwargs
            )
        )

        prev_state_var = tf.convert_to_tensor(self.prev_state_var)
        prev_state_var.set_shape((None, self.state_dim))

        l_step_input = L.InputLayer(
            shape=(None, self.embedding_dim),
            input_var=flat_embedding_var,
        )
        l_step_prev_state = L.InputLayer(
            shape=(None, self.state_dim),
            input_var=prev_state_var
        )
        step_hidden_var = L.get_output(
            self.l_hidden.get_step_layer(
                l_step_input,
                prev_state_layer=l_step_prev_state,
            ), **kwargs
        )

        action_var = L.get_output(
            self.l_flat_action,
            {self.l_flat_hidden: step_hidden_var}
        )

        update_ops = [
            tf.assign(self.prev_state_var, step_hidden_var),
        ]

        with tf.control_dependencies(update_ops):
            action_var = tf.identity(action_var)

        return Categorical(dim=self.action_dim).sample_sym(dict(prob=action_var))

    def get_output(self, obs_var, **kwargs):
        return L.get_output(
            self.l_action,
            {
                self.l_obs_input: obs_var,
            },
            **kwargs
        )

    @property
    def recurrent(self):
        return True


class Net(object):
    def new_networks(self, env_spec):
        obs_embedding_network = ObsEmbeddingNetwork(env_spec=env_spec)
        cell = GRUCell(num_units=100, activation=tf.nn.relu, weight_normalization=True)
        action_network = ActionNetwork(
            env_spec=env_spec,
            obs_embedding_network=obs_embedding_network,
            cell=cell,
        )
        return action_network,
