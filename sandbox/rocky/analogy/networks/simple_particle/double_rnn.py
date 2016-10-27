import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np

from sandbox.rocky.tf.misc import tensor_utils


class ImageObsEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        assert len(env_spec.observation_space.components) == 3
        obs_dim = env_spec.observation_space.flat_dim
        pos_obs_dim = sum([x.flat_dim for x in env_spec.observation_space.components[:2]])
        image_obs_dim = env_spec.observation_space.components[-1].flat_dim
        image_shape = env_spec.observation_space.components[-1].shape

        l_obs_input = L.InputLayer(
            (None, None, obs_dim),
            name="obs_input"
        )
        l_flat_obs_input = L.TemporalFlattenLayer(l_obs_input)

        l_pos_obs_input = L.SliceLayer(
            l_flat_obs_input,
            indices=slice(pos_obs_dim),
            axis=1
        )
        l_image_obs_input = L.SliceLayer(
            l_flat_obs_input,
            indices=slice(pos_obs_dim, obs_dim),
            axis=1
        )

        ### Begin image layers
        l_image = L.reshape(
            l_image_obs_input,
            ([0],) + image_shape,
        )

        l_image = L.Conv2DLayer(
            l_image,
            num_filters=20,
            filter_size=3,
            stride=(1, 1),
            pad='SAME',
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )
        l_image = L.Conv2DLayer(
            l_image,
            num_filters=20,
            filter_size=3,
            stride=(1, 1),
            pad='SAME',
            nonlinearity=None,
            weight_normalization=True
        )
        l_image = L.SpatialExpectedSoftmaxLayer(
            l_image,
        )
        ### End image layers

        ### Begin pos layers
        l_pos = l_pos_obs_input
        l_pos = L.DenseLayer(
            l_pos,
            num_units=100,
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )
        l_pos = L.DenseLayer(
            l_pos,
            num_units=100,
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )
        ### End pos layers

        l_flat_embedding = L.concat([
            l_image,
            l_pos,
        ])

        l_embedding = L.TemporalUnflattenLayer(l_flat_embedding, l_obs_input)

        self.embedding_dim = l_embedding.output_shape[-1]
        self.input_vars = [l_obs_input.input_var]
        self.l_obs_input = l_obs_input
        self.l_flat_embedding = l_flat_embedding
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


class StateObsEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        obs_dim = env_spec.observation_space.flat_dim

        l_obs_input = L.InputLayer(
            (None, None, obs_dim),
            name="obs_input"
        )

        l_embedding = L.TemporalUnflattenLayer(
            L.DenseLayer(
                L.TemporalFlattenLayer(l_obs_input),
                num_units=100,
                nonlinearity=tf.nn.relu,
            ),
            ref_layer=l_obs_input
        )

        # l_embedding = L.IdentityLayer(l_obs_input)

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


class SummaryNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, obs_embedding_network, state_dim=100):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec

        l_embedding = obs_embedding_network.l_embedding

        l_summary = L.GRULayer(
            l_embedding,
            num_units=state_dim,
            hidden_nonlinearity=tf.nn.relu,
            weight_normalization=True,
        )

        summary_var = tf.Variable(
            initial_value=np.zeros((0, state_dim), dtype=np.float32),
            validate_shape=False,
            name="summary",
            trainable=False
        )

        self.l_embedding = l_embedding
        self.l_summary = l_summary
        self.l_obs_input = obs_embedding_network.l_obs_input
        self.summary_var = summary_var
        self.output_dim = state_dim
        self.state_dim = state_dim
        self.embedding_dim = obs_embedding_network.embedding_dim
        self.input_vars = obs_embedding_network.input_vars
        self.output_layer = l_summary
        LayersPowered.__init__(self, [l_summary])

    def get_update_op(self, obs_var, valids_var, **kwargs):
        summary = self.get_output(obs_var=obs_var, valids_var=valids_var, **kwargs)
        return tf.assign(self.summary_var, summary, validate_shape=False)

    # def get_params_internal(self, **tags):
    #     params = LayersPowered.get_params_internal(self, **tags)
    #     if not tags.get('trainable', False):
    #         params.append(self.summary_var)
    #     return params

    def get_output(self, obs_var, valids_var, **kwargs):
        # we shift and pad the action into prev action, so that it aligns with the input of the action network
        # hence effectively, the last action is ignored

        all_summary = L.get_output(
            self.l_summary,
            {self.l_obs_input: obs_var}
        )

        batch_size = tf.shape(obs_var)[0]

        last_ids = tf.cast(tf.reduce_sum(valids_var, -1) - 1, tf.int32)

        last_summary = tensor_utils.fancy_index_sym(all_summary, [tf.range(batch_size), last_ids])

        return last_summary


class ActionNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, obs_embedding_network, summary_network, state_dim=100):
        Serializable.quick_init(self, locals())
        action_dim = env_spec.action_space.flat_dim

        l_embedding = obs_embedding_network.l_embedding

        summary_state_dim = summary_network.state_dim

        l_summary_input = L.InputLayer(
            shape=(None, None, summary_state_dim),
        )

        l_hidden = L.GRULayer(
            L.concat([l_embedding, l_summary_input], axis=2),
            num_units=state_dim,
            hidden_nonlinearity=tf.nn.relu,
            weight_normalization=True,
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
            nonlinearity=None
        )

        l_action = L.TemporalUnflattenLayer(l_flat_action, obs_embedding_network.l_obs_input)

        self.l_obs_input = obs_embedding_network.l_obs_input

        self.prev_state_var = tf.Variable(
            initial_value=np.zeros((0, state_dim), dtype=np.float32),
            validate_shape=False,
            name="prev_state",
            trainable=False
        )
        self.l_embedding = l_embedding
        self.embedding_dim = obs_embedding_network.embedding_dim
        self.summary_network = summary_network
        self.l_summary_input = l_summary_input
        self.l_hidden = l_hidden
        self.l_action = l_action
        self.l_flat_hidden = l_flat_hidden
        self.l_flat_action = l_flat_action
        self.output_layer = l_action
        self.state_dim = state_dim
        self.summary_state_dim = summary_state_dim
        self.action_dim = action_dim
        LayersPowered.__init__(self, [l_action])

    # def get_params_internal(self, **tags):
    #     params = LayersPowered.get_params_internal(self, **tags)
    #     if not tags.get('trainable', False):
    #         params.append(self.prev_state_var)
    #     return params

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

        l_step_input = L.InputLayer(
            shape=(None, self.embedding_dim + self.summary_state_dim),
            input_var=tf.concat(1, [flat_embedding_var, self.summary_network.summary_var]),
        )
        l_step_prev_state = L.InputLayer(
            shape=(None, self.state_dim),
            input_var=self.prev_state_var
        )
        step_hidden_var = L.get_output(
            self.l_hidden.get_step_layer(
                l_in=l_step_input,
                l_prev_hidden=l_step_prev_state,
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

        return action_var

    def get_output(self, obs_var, summary_var, **kwargs):
        return L.get_output(
            self.l_action,
            {
                self.l_obs_input: obs_var,
                self.l_summary_input: tensor_utils.temporal_tile_sym(summary_var, ref_var=obs_var),
            },
            **kwargs
        )

    @property
    def recurrent(self):
        return True


class Net(object):
    def __init__(self, obs_type='full_state'):
        self.obs_type = obs_type

    def new_networks(self, env_spec):
        if self.obs_type == 'full_state':
            obs_embedding_network = StateObsEmbeddingNetwork(env_spec=env_spec)
        elif self.obs_type == 'image':
            obs_embedding_network = ImageObsEmbeddingNetwork(env_spec=env_spec)
        else:
            raise NotImplementedError
        summary_network = SummaryNetwork(
            env_spec=env_spec,
            obs_embedding_network=obs_embedding_network,
        )
        action_network = ActionNetwork(
            env_spec=env_spec,
            summary_network=summary_network,
            obs_embedding_network=obs_embedding_network,
        )
        return summary_network, action_network
