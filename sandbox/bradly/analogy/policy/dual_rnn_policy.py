# We use one RNN to extract the goal after watching the demo and another RNN for the policy itself
# (we need memory to remember which points we've reached in the past).


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


class DualRNNAnalogyPolicy(AnalogyPolicy, LayersPowered, Serializable):
    def __init__(self, env_spec, name, rnn_hidden_size=32, rnn_hidden_nonlinearity=tf.nn.tanh,
                 state_include_action=False,
                 network_type=rnn_utils.NetworkType.GRU,  # mlp_hidden_nonlinearity=tf.nn.tanh,
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
                embedding_dim = 40  # 100

                embedding_network = MLP(
                    name='embedding_network',
                    input_shape=obs_input_shape,
                    input_layer=l_flat_obs_input,
                    output_dim=embedding_dim,
                    hidden_sizes=(),
                    hidden_nonlinearity=tf.nn.relu,
                    output_nonlinearity=tf.nn.softmax,
                    weight_normalization=weight_normalization,
                    batch_normalization=batch_normalization

                )

            # action_network = MLP(
            #     name="action_network",
            #     input_shape=(mlp_input_dim,),
            #     input_layer=L.concat([l_flat_embedding, l_summary_in], axis=1),
            #     hidden_sizes=mlp_hidden_sizes,
            #     hidden_nonlinearity=mlp_hidden_nonlinearity,
            #     output_dim=action_dim,
            #     output_nonlinearity=output_nonlinearity,
            #     weight_normalization=weight_normalization,
            #     # layer_normalization=layer_normalization,
            #     batch_normalization=batch_normalization,
            # )

                #embedding_network = ConvNetwork(
                #    name="embedding_network",
                #    input_shape=obs_input_shape,
                #    input_layer=l_flat_obs_input,
                #    output_dim=embedding_dim,
                #    hidden_sizes=(),
                #    conv_filters=(embedding_dim, embedding_dim // 2),
                #    conv_filter_sizes=(5, 3),
                #    conv_strides=(1, 1),
                #    conv_pads=('SAME', 'SAME'),
                #    hidden_nonlinearity=tf.nn.relu,
                #    output_nonlinearity=L.spatial_expected_softmax,
                #    weight_normalization=weight_normalization,
                #    # layer_normalization=layer_normalization,
                #    batch_normalization=batch_normalization,
                #    # batch_normalization=True,
                #)

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

            l_flat_summary_in = L.InputLayer(
                shape=(None, rnn_hidden_size),
                name="flat_summary_in",
                input_var=summary_var
            )
            l_summary_in = L.OpLayer(
                l_flat_summary_in,
                extras=[l_embedding],
                name="summary_in",
                op=lambda flat_summary, embedding: tf.tile(
                    tf.expand_dims(flat_summary, 1),
                    tf.pack([1, tf.shape(embedding)[1], 1])
                ),
                shape_op=lambda flat_summary_shape, embedding_shape: (flat_summary_shape[0], embedding_shape[1],
                                                                      flat_summary_shape[1])
            )

            self.summary = np.zeros((1, rnn_hidden_size), dtype=np.float32)


            # mlp_input_dim = embedding_dim + rnn_hidden_size
            action_network = rnn_utils.create_recurrent_network(
                network_type,
                input_shape=(embedding_dim + rnn_hidden_size,),
                input_layer=L.concat([l_embedding, l_summary_in], axis=2),
                output_dim=action_dim,
                hidden_dim=rnn_hidden_size,
                hidden_nonlinearity=rnn_hidden_nonlinearity,
                output_nonlinearity=output_nonlinearity,
                name="action_network",
                weight_normalization=weight_normalization,
                layer_normalization=layer_normalization,
                **network_args
            )

            # action_network = MLP(
            #     name="action_network",
            #     input_shape=(mlp_input_dim,),
            #     input_layer=L.concat([l_flat_embedding, l_summary_in], axis=1),
            #     hidden_sizes=mlp_hidden_sizes,
            #     hidden_nonlinearity=mlp_hidden_nonlinearity,
            #     output_dim=action_dim,
            #     output_nonlinearity=output_nonlinearity,
            #     weight_normalization=weight_normalization,
            #     # layer_normalization=layer_normalization,
            #     batch_normalization=batch_normalization,
            # )

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

            self.prev_actions = None
            self.prev_state = None

            if state_include_action:
                summary_inputs = [l_obs_input.input_var, l_action_input.input_var]
            else:
                summary_inputs = [l_obs_input.input_var]

            self.f_update_summary = tensor_utils.compile_function(
                summary_inputs,
                tf.assign(summary_var, L.get_output(l_summary, phase='test')),
            )

            self.f_compute_summary = tensor_utils.compile_function(
                summary_inputs,
                L.get_output(l_summary, phase='test'),
                # tf.assign(summary_var, , validate_shape=False),
            )

            flat_embedding_var = L.get_output(l_flat_embedding, {l_obs_input: tf.expand_dims(obs_var, 0)})

            # l_flat_embedding

            self.f_action = tensor_utils.compile_function(
                [obs_var, action_network.step_prev_state_layer.input_var],
                L.get_output(
                    [
                        action_network.step_output_layer,
                        action_network.step_state_layer,
                    ],
                    {action_network.step_input_layer: tf.concat(1, [flat_embedding_var, self.summary_var])},
                    phase='test'
                ),
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

        # import ipdb; ipdb.set_trace()

        batch_size = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        # flat_obs_var = tf.reshape(obs_var, (-1,) + self.obs_input_shape)
        summary_var = tf.tile(
            tf.expand_dims(summary_var, 1),
            tf.pack([1, n_steps, 1]),
        )
        action_var = L.get_output(
            self.action_network.output_layer, {
                self.l_obs_input: obs_var,
                self.l_summary_in: summary_var
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

    def apply_demos(self, paths):

        # if len(dones) != len(self.summary):
        #     self.summary = np.zeros((len(dones), self.summary_network.hidden_dim))

        # if np.any(dones):
        #     done_paths = [p for done, p in zip(dones, paths) if done]

        max_len = np.max([len(p["rewards"]) for p in paths])

        demo_obs = [p["observations"] for p in paths]
        demo_obs = np.asarray([tensor_utils.pad_tensor(o, max_len) for o in demo_obs])

        demo_actions = [p["actions"] for p in paths]
        demo_actions = np.asarray([tensor_utils.pad_tensor(a, max_len) for a in demo_actions])

        demo_valids = [np.ones_like(p["rewards"]) for p in paths]
        demo_valids = np.asarray([tensor_utils.pad_tensor(v, max_len) for v in demo_valids])

        assert np.all(demo_valids)

        self.summary = self.f_compute_summary(demo_obs, demo_actions)
            # self.summary[dones] = new_summary

            # import ipdb; ipdb.set_trace()
            #
            # # self.f_update_summary

            # run through paths, but only until it's valid?
            # how to express this?
            # we can sum over all valids

            # self.f_update_summary(demo_obs, demo_actions)

    def reset(self, dones=None):
        # self.prev_actions = None
        self.prev_state = self.action_network.state_init_param.eval()

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        actions, new_states = self.f_action([flat_obs], [self.prev_state])
        self.prev_state = new_states[0]
        return actions[0], dict()

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self.f_action(flat_obs, self.summary)
        return actions, dict()
