from rllab.core.serializable import Serializable
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import GRUNetwork, MLP, ConvNetwork, ConvMergeNetwork
from sandbox.rocky.tf.misc import tensor_utils
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np

from sandbox.rocky.tf.policies import rnn_utils
from sandbox.rocky.tf.spaces.box import Box


class DemoRNNMLPAnalogyPolicy(AnalogyPolicy, LayersPowered, Serializable):
    def __init__(self, env_spec, name, rnn_hidden_size=32, rnn_hidden_nonlinearity=tf.nn.tanh,
                 mlp_hidden_sizes=(32, 32), state_include_action=False,
                 network_type=rnn_utils.NetworkType.GRU, mlp_hidden_nonlinearity=tf.nn.tanh,
                 weight_normalization=False, layer_normalization=False, batch_normalization=False,
                 output_nonlinearity=None, network_args=None):
        Serializable.quick_init(self, locals())
        with tf.variable_scope(name):
            AnalogyPolicy.__init__(self, env_spec=env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            use_embedding = isinstance(env_spec.observation_space, Box) and len(env_spec.observation_space.shape) == 3
            if use_embedding:
                obs_input_shape = env_spec.observation_space.shape
            else:
                obs_input_shape = (env_spec.observation_space.flat_dim,)

            if network_args is None:
                network_args = dict()

            l_obs_input = L.InputLayer(
                shape=(None, None, obs_dim),
                name="obs_input"
            )
            l_action_input = L.InputLayer(
                shape=(None, None, action_dim),
                name="action_input"
            )

            l_flat_obs_input = L.OpLayer(
                l_obs_input,
                op=lambda obs_input: tf.reshape(obs_input, (-1,) + obs_input_shape),
                shape_op=lambda shape: (None,) + obs_input_shape
            )

            if use_embedding:
                assert state_include_action
                embedding_dim = 40#100

                embedding_network = ConvNetwork(
                    name="embedding_network",
                    input_shape=obs_input_shape,
                    input_layer=l_flat_obs_input,
                    output_dim=embedding_dim,
                    hidden_sizes=(),
                    conv_filters=(embedding_dim // 2, embedding_dim // 2),
                    conv_filter_sizes=(3, 3),
                    conv_strides=(1, 1),
                    conv_pads=('SAME', 'SAME'),
                    hidden_nonlinearity=tf.nn.relu,
                    output_nonlinearity=L.spatial_expected_softmax,
                    weight_normalization=weight_normalization,
                    # layer_normalization=layer_normalization,
                    batch_normalization=batch_normalization,
                    # batch_normalization=True,
                )

                l_flat_embedding = embedding_network.output_layer
                l_embedding = L.OpLayer(
                    l_flat_embedding,
                    extras=[l_obs_input],
                    name="reshape_feature",
                    op=lambda flat_embedding, input: tf.reshape(
                        flat_embedding,
                        tf.pack([tf.shape(input)[0], tf.shape(input)[1], embedding_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], embedding_dim)
                )

            else:
                l_embedding = l_obs_input
                l_flat_embedding = l_flat_obs_input
                embedding_dim = obs_dim

            if state_include_action:
                l_gru_input = L.concat([l_embedding, l_action_input], axis=2)
                gru_input_dim = embedding_dim + action_dim
            else:
                l_gru_input = l_embedding
                gru_input_dim = embedding_dim

            summary_network = rnn_utils.create_recurrent_network(
                network_type,
                input_shape=(gru_input_dim,),
                input_layer=l_gru_input,
                output_dim=rnn_hidden_size,
                hidden_dim=rnn_hidden_size,
                hidden_nonlinearity=rnn_hidden_nonlinearity,
                output_nonlinearity=None,
                name="summary_network",
                weight_normalization=weight_normalization,
                layer_normalization=layer_normalization,
                # batch_normalization=batch_normalization,
                **network_args,
            )

            summary_var = tf.Variable(initial_value=np.zeros((1, rnn_hidden_size), dtype=np.float32), trainable=False,
                                      name="summary")

            obs_var = env_spec.observation_space.new_tensor_variable("obs", extra_dims=1)

            l_summary_in = L.InputLayer(
                shape=(None, rnn_hidden_size),
                name="summary_in",
                input_var=summary_var
            )

            mlp_input_dim = embedding_dim + rnn_hidden_size
            action_network = MLP(
                name="action_network",
                input_shape=(mlp_input_dim,),
                input_layer=L.concat([l_flat_embedding, l_summary_in], axis=1),
                hidden_sizes=mlp_hidden_sizes,
                hidden_nonlinearity=mlp_hidden_nonlinearity,
                output_dim=action_dim,
                output_nonlinearity=output_nonlinearity,
                weight_normalization=weight_normalization,
                # layer_normalization=layer_normalization,
                batch_normalization=batch_normalization,
            )

            l_summary = L.SliceLayer(summary_network.recurrent_layer, indices=-1, axis=1)

            self.summary_network = summary_network
            self.action_network = action_network
            self.l_summary = l_summary
            self.l_embedding = l_embedding
            self.l_obs_input = l_obs_input
            self.l_flat_obs_input = l_flat_obs_input
            self.l_action_input = l_action_input
            self.l_summary_in = l_summary_in
            self.l_summary_input = summary_network.input_layer
            self.state_include_action = state_include_action
            self.summary_var = summary_var
            self.obs_input_shape = obs_input_shape

            if state_include_action:
                summary_inputs = [l_obs_input.input_var, l_action_input.input_var]
            else:
                summary_inputs = [l_obs_input.input_var]

            self.f_update_summary = tensor_utils.compile_function(
                summary_inputs,
                tf.assign(summary_var, L.get_output(l_summary, phase='test')),
            )

            self.f_action = tensor_utils.compile_function(
                [obs_var],
                L.get_output(action_network.output_layer, {
                    l_flat_obs_input: tf.reshape(obs_var, (-1,) + obs_input_shape),
                }, phase='test'),
            )

            self.gru_size = rnn_hidden_size

            LayersPowered.__init__(self, [summary_network.output_layer, action_network.output_layer])

    def action_sym(self, obs_var, state_info_vars, **kwargs):
        demo_obs_var = state_info_vars["demo_obs"]
        demo_action_var = state_info_vars["demo_action"]

        if self.state_include_action:
            summary_inputs = {self.l_obs_input: demo_obs_var, self.l_action_input: demo_action_var}
        else:
            summary_inputs = {self.l_obs_input: demo_obs_var}
        summary_var = L.get_output(self.l_summary, summary_inputs, **kwargs)

        batch_size = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        flat_obs_var = tf.reshape(obs_var, (-1,) + self.obs_input_shape)
        flat_summary_var = tf.reshape(
            tf.tile(
                tf.expand_dims(summary_var, 1),
                tf.pack([1, n_steps, 1]),
            ),
            (-1, self.gru_size),
        )
        action_var = L.get_output(
            self.action_network.output_layer, {
                self.l_flat_obs_input: flat_obs_var,
                self.l_summary_in: flat_summary_var
            }, **kwargs
        )

        return tf.reshape(action_var, tf.pack([batch_size, n_steps, self.action_space.flat_dim]))

    def apply_demo(self, path):
        demo_obs = path["observations"]
        demo_actions = path["actions"]
        if self.state_include_action:
            summary_inputs = [[demo_obs], [demo_actions]]
        else:
            summary_inputs = [[demo_obs]]
        self.f_update_summary(*summary_inputs)

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self.f_action([flat_obs])
        return action[0], dict()
