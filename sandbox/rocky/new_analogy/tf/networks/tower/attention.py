import sandbox.rocky.tf.core.layers as L
import sandbox.rocky.analogy.core.layers as LL
from sandbox.rocky.analogy.rnn_cells import GRUCell, AttentionCell
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces import Discrete, Box


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
            obs_var = self.l_obs_input.input_var
        return L.get_output(
            self.l_embedding,
            {self.l_obs_input: obs_var},
            **kwargs
        )


class ActionEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        action_dim = env_spec.action_space.flat_dim

        l_action_input = L.InputLayer(
            (None, None, action_dim),
            name="action_input"
        )

        l_embedding = LL.TemporalDenseLayer(
            l_action_input,
            num_units=100,
            nonlinearity=tf.nn.relu,
        )

        self.input_vars = [l_action_input.input_var]
        self.l_action_input = l_action_input
        self.embedding_dim = l_embedding.output_shape[-1]
        self.l_embedding = l_embedding
        LayersPowered.__init__(self, [l_embedding])

    def get_output(self, action_var=None, **kwargs):
        if action_var is None:
            action_var = self.l_action_input.input_var
        return L.get_output(
            self.l_embedding,
            {self.l_action_input: action_var},
            **kwargs
        )


class SummaryNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, obs_embedding_network, action_embedding_network, cell):
        self.env_spec = env_spec

        l_obs_embedding = obs_embedding_network.l_embedding
        l_obs_input = obs_embedding_network.l_obs_input
        l_action_embedding = action_embedding_network.l_embedding
        l_action_input = action_embedding_network.l_action_input

        l_embedding = L.concat([l_obs_embedding, l_action_embedding], axis=2)

        l_valid_input = L.InputLayer(
            (None, None),
            name="valid_input"
        )

        l_forward_embedding = LL.TfRNNLayer(
            l_embedding,
            cell=cell,
        )

        l_backward_embedding = LL.TemporalReverseLayer(
            LL.TfRNNLayer(
                LL.TemporalReverseLayer(
                    l_embedding,
                    valid_layer=l_valid_input
                ),
                cell=cell
            ),
            valid_layer=l_valid_input,
        )

        l_summary = L.ElemwiseSumLayer([
            LL.TemporalDenseLayer(
                l_forward_embedding,
                num_units=cell.output_size,
                nonlinearity=tf.nn.relu,
                weight_normalization=True,
            ),
            LL.TemporalDenseLayer(
                l_backward_embedding,
                num_units=cell.output_size,
                nonlinearity=tf.nn.relu,
                weight_normalization=True,
                b=None,
            ),
        ])

        summary_var = tf.Variable(
            initial_value=np.zeros((0, 0, l_summary.output_shape[-1]), dtype=np.float32),
            validate_shape=False,
            name="summary",
            trainable=False
        )

        self.output_dim = cell.output_size
        self.l_summary = l_summary
        self.l_obs_input = l_obs_input
        self.l_action_input = l_action_input
        self.l_valid_input = l_valid_input
        self.summary_var = summary_var
        self.embedding_dim = l_embedding.output_shape[-1]
        self.input_vars = obs_embedding_network.input_vars + action_embedding_network.input_vars
        self.output_layer = l_summary
        LayersPowered.__init__(self, [l_summary])

    def get_update_op(self, obs_var, actions_var, valids_var, **kwargs):
        summary = self.get_output(obs_var=obs_var, actions_var=actions_var, valids_var=valids_var, **kwargs)
        return tf.assign(self.summary_var, summary, validate_shape=False)

    def get_output(self, obs_var, actions_var, valids_var, **kwargs):
        summary = L.get_output(
            self.l_summary,
            {self.l_obs_input: obs_var, self.l_action_input: actions_var, self.l_valid_input: valids_var},
            **kwargs
        )
        return summary


class ActionNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, obs_embedding_network, summary_network, cell, attention_cell_cls):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec

        action_dim = env_spec.action_space.flat_dim

        l_embedding = obs_embedding_network.l_embedding

        summary_output_dim = summary_network.output_dim

        l_summary_input = L.InputLayer(
            shape=(None, None, summary_network.output_dim),
        )

        l_hidden = LL.AttentionLayer(
            l_embedding,
            attend_layer=l_summary_input,
            valid_layer=summary_network.l_valid_input,
            cell=cell,
            attention_cell_cls=attention_cell_cls,
            attention_vec_size=100,
            num_heads=2
        )

        l_flat_hidden = L.TemporalFlattenLayer(l_hidden)

        l_action_hid = L.DenseLayer(
            l_flat_hidden,
            num_units=100,
            nonlinearity=tf.nn.relu,
        )

        assert isinstance(self.env_spec.action_space, Box)
        l_flat_action = L.DenseLayer(
            l_action_hid,
            num_units=action_dim,
        )

        l_action = L.TemporalUnflattenLayer(l_flat_action, obs_embedding_network.l_obs_input)

        self.l_obs_input = obs_embedding_network.l_obs_input

        self.prev_state_var = tf.Variable(
            initial_value=np.zeros((0, l_hidden.state_dim), dtype=np.float32),
            validate_shape=False,
            name="prev_state",
            trainable=False
        )
        self.l_embedding = l_embedding
        self.embedding_dim = obs_embedding_network.embedding_dim
        self.summary_network = summary_network
        self.l_summary_input = l_summary_input
        self.l_hidden = l_hidden
        self.l_flat_hidden = l_flat_hidden
        self.l_action = l_action
        self.l_flat_action = l_flat_action
        self.output_layer = l_action
        self.state_dim = l_hidden.state_dim
        self.summary_output_dim = summary_output_dim
        self.action_dim = action_dim
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

    def get_step_op(self, obs_var, demo_valids_var, **kwargs):
        flat_embedding_var = tensor_utils.temporal_flatten_sym(
            L.get_output(
                self.l_embedding,
                {
                    self.l_obs_input: tf.expand_dims(obs_var, 1),
                }, **kwargs
            )
        )

        l_step_input = L.InputLayer(
            shape=(None, self.embedding_dim),
            input_var=flat_embedding_var,
        )
        summary_var = tf.convert_to_tensor(self.summary_network.summary_var)
        summary_var.set_shape((None, None, self.summary_output_dim))
        l_step_attend = L.InputLayer(
            shape=(None, None, self.summary_output_dim),
            input_var=summary_var,
        )
        prev_state_var = tf.convert_to_tensor(self.prev_state_var)
        prev_state_var.set_shape((None, self.l_hidden.state_dim))
        l_step_prev_state = L.InputLayer(
            shape=(None, self.state_dim),
            input_var=prev_state_var
        )
        l_step_demo_valids = L.InputLayer(
            shape=(None, None),
            input_var=demo_valids_var,
        )

        rnn_step_layer = self.l_hidden.get_step_layer(
            incoming=[l_step_input, l_step_attend, l_step_demo_valids],
            prev_state_layer=l_step_prev_state,
        )

        recurrent_state_output = dict()

        step_hidden_var = L.get_output(
            rnn_step_layer,
            recurrent_state_output=recurrent_state_output,
            **kwargs
        )
        step_state_var = recurrent_state_output[rnn_step_layer]

        action_var = L.get_output(
            self.l_flat_action,
            {self.l_flat_hidden: step_hidden_var}
        )

        update_ops = [
            tf.assign(self.prev_state_var, step_state_var),
        ]

        with tf.control_dependencies(update_ops):
            action_var = tf.identity(action_var)

        assert isinstance(self.env_spec.action_space, Box)
        return action_var#Categorical(dim=self.action_dim).sample_sym(dict(prob=action_var))

    def get_output(self, obs_var, summary_var, demo_valids_var, **kwargs):
        return L.get_output(
            self.l_action,
            {
                self.l_obs_input: obs_var,
                self.l_summary_input: summary_var,
                self.summary_network.l_valid_input: demo_valids_var,
            },
            **kwargs
        )

    @property
    def recurrent(self):
        return True


class Net(object):
    def __init__(self, obs_type='full_state', cell=None, attention_cell_cls=None):
        self.obs_type = obs_type
        if cell is None:
            cell = GRUCell(num_units=100, activation=tf.nn.relu, weight_normalization=True)
        if attention_cell_cls is None:
            attention_cell_cls = AttentionCell
        self.cell = cell
        self.attention_cell_cls = attention_cell_cls

    def new_networks(self, env_spec):
        obs_embedding_network = ObsEmbeddingNetwork(env_spec=env_spec)
        action_embedding_network = ActionEmbeddingNetwork(env_spec=env_spec)

        summary_network = SummaryNetwork(
            env_spec=env_spec,
            obs_embedding_network=obs_embedding_network,
            action_embedding_network=action_embedding_network,
            cell=self.cell
        )
        action_network = ActionNetwork(
            env_spec=env_spec,
            obs_embedding_network=obs_embedding_network,
            summary_network=summary_network,
            cell=self.cell,
            attention_cell_cls=self.attention_cell_cls,
        )
        return summary_network, action_network
