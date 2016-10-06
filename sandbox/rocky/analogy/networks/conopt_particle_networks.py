import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
import tensorflow as tf


def flatten_sym(var):
    return tf.reshape(var, tf.pack([-1, tf.shape(var)[2]]))


def unflatten_sym(var, ref_var):
    return tf.reshape(
        var,
        tf.pack([tf.shape(ref_var)[0], tf.shape(ref_var)[1], tf.shape(var)[1]])
    )


def unflatten_layer(layer, ref_layer):
    return L.OpLayer(
        layer,
        extras=[ref_layer],
        op=unflatten_sym,
        shape_op=lambda flat_shape, input_shape: (input_shape[0], input_shape[1], flat_shape[1])
    )


def flatten_layer(layer):
    return L.OpLayer(
        layer,
        op=flatten_sym,
        shape_op=lambda shape: (None, shape[2]),
    )


class ImageObsEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        obs_dim = env_spec.observation_space.flat_dim
        pos_obs_dim = sum([x.flat_dim for x in env_spec.observation_space.components[:2]])
        image_obs_dim = env_spec.observation_space.components[-1].flat_dim

        l_obs_input = L.InputLayer(
            (None, None, obs_dim),
            name="obs_input"
        )
        l_flat_obs_input = flatten_layer(l_obs_input)

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
        l_image = l_image_obs_input

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

        l_embedding = unflatten_layer(l_flat_embedding, l_obs_input)

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
        l_flat_obs_input = flatten_layer(l_obs_input)

        l_hidden = L.DenseLayer(
            l_flat_obs_input,
            num_units=100,
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )
        l_flat_embedding = L.DenseLayer(
            l_hidden,
            num_units=100,
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )

        l_embedding = unflatten_layer(l_flat_embedding, l_obs_input)

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


class ActionEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        action_dim = env_spec.action_space.flat_dim
        l_action_input = L.InputLayer(
            (None, None, action_dim)
        )
        l_flat_action_input = flatten_layer(l_action_input)

        ### Begin action layers
        l_action = l_flat_action_input
        l_action = L.DenseLayer(
            l_action,
            num_units=100,
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )
        l_action = L.DenseLayer(
            l_action,
            num_units=100,
            nonlinearity=tf.nn.relu,
            weight_normalization=True
        )
        ### End action layers

        l_embedding = unflatten_layer(l_action, l_action_input)

        self.input_vars = [l_action_input.input_var]
        self.l_action_input = l_action_input
        self.l_flat_embedding = l_action
        self.l_embedding = l_embedding
        LayersPowered.__init__(self, [l_embedding])

    def get_output(self, action_var=None, **kwargs):
        if action_var is None:
            action_var = self.l_action_input
        return L.get_output(
            self.l_embedding,
            {self.l_action_input: action_var},
            **kwargs
        )


class SummaryNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, obs_embedding_network, action_embedding_network):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec

        l_embedding = L.concat([
            obs_embedding_network.l_embedding,
            action_embedding_network.l_embedding
        ], axis=2)

        l_all_summary = L.GRULayer(
            l_embedding,
            num_units=100,
            hidden_nonlinearity=tf.nn.relu,
            weight_normalization=True,
        )

        l_summary = L.SliceLayer(
            l_all_summary,
            indices=-1,
            axis=1
        )

        self.l_summary = l_summary
        self.l_obs_input = obs_embedding_network.l_obs_input
        self.l_action_input = action_embedding_network.l_action_input
        self.output_dim = l_summary.output_shape[-1]
        self.input_vars = obs_embedding_network.input_vars + action_embedding_network.input_vars
        self.output_layer = l_summary
        LayersPowered.__init__(self, [l_summary])

    def get_output(self, obs_var=None, action_var=None, **kwargs):
        if obs_var is None:
            obs_var = self.l_obs_input.input_var
        if action_var is None:
            action_var = self.l_action_input.input_var
        return L.get_output(
            self.l_summary,
            {self.l_obs_input: obs_var, self.l_action_input: action_var},
            **kwargs
        )


class ActionNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, summary_network, obs_embedding_network):
        Serializable.quick_init(self, locals())
        action_dim = env_spec.action_space.flat_dim

        l_flat_obs_embedding = obs_embedding_network.l_flat_embedding
        l_summary_input = L.InputLayer(
            (None, summary_network.output_dim),
        )

        l_embedding = L.concat([
            l_flat_obs_embedding,
            l_summary_input,
        ], axis=1)

        l_hidden = L.DenseLayer(
            l_embedding,
            num_units=100,
            nonlinearity=tf.nn.relu,
        )

        l_flat_action = L.DenseLayer(
            l_hidden,
            num_units=action_dim,
            nonlinearity=None,
        )

        l_obs_input = obs_embedding_network.l_obs_input

        l_action = unflatten_layer(l_flat_action, ref_layer=l_obs_input)

        self.l_action = l_action
        self.obs_embedding_network = obs_embedding_network
        self.l_obs_input = l_obs_input
        self.l_summary_input = l_summary_input
        self.summary_network = summary_network
        self.input_vars = [l_obs_input.input_var, l_summary_input.input_var]
        self.output_layer = l_action

        LayersPowered.__init__(self, [l_action])

    def get_output(self, obs_var=None, summary_var=None, **kwargs):
        if obs_var is None:
            obs_var = self.l_obs_input.input_var
        if summary_var is None:
            summary_var = self.l_summary_input.input_var
        n_steps = tf.shape(obs_var)[1]
        repeated_summary_var = tf.reshape(
            tf.tile(
                tf.expand_dims(summary_var, 1),
                tf.pack([1, n_steps, 1]),
            ),
            (-1, self.summary_network.output_dim),
        )

        return L.get_output(
            self.l_action,
            {self.l_obs_input: obs_var, self.l_summary_input: repeated_summary_var},
            **kwargs
        )

    def get_flat_output(self, obs_var=None, summary_var=None, *args, **kwargs):
        if obs_var is None:
            obs_var = self.l_obs_input.input_var
        return flatten_sym(
            self.get_output(
                obs_var=tf.expand_dims(obs_var, 1),
                summary_var=summary_var,
                **kwargs
            )
        )
