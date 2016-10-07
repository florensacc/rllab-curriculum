import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
import tensorflow as tf
import numpy as np


class StateObsEmbeddingNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec):
        Serializable.quick_init(self, locals())
        obs_dim = env_spec.observation_space.flat_dim

        l_obs_input = L.InputLayer(
            (None, None, obs_dim),
            name="obs_input"
        )

        l_embedding = L.IdentityLayer(l_obs_input)

        self.input_vars = [l_obs_input.input_var]
        self.l_obs_input = l_obs_input
        self.embedding_dim = obs_dim
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


# class ActionEmbeddingNetwork(LayersPowered, Serializable):
#     def __init__(self, env_spec):
#         Serializable.quick_init(self, locals())
#         action_dim = env_spec.action_space.flat_dim
#         l_action_input = L.InputLayer(
#             (None, None, action_dim)
#         )
#
#         l_embedding = L.IdentityLayer(l_action_input)
#
#         self.input_vars = [l_action_input.input_var]
#         self.l_action_input = l_action_input
#         self.embedding_dim = action_dim
#         self.l_embedding = l_embedding
#         LayersPowered.__init__(self, [l_embedding])
#
#     def get_output(self, action_var=None, **kwargs):
#         if action_var is None:
#             action_var = self.l_action_input
#         return L.get_output(
#             self.l_embedding,
#             {self.l_action_input: action_var},
#             **kwargs
#         )


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
            layer_normalization=True,
        )

        l_last_summary = L.SliceLayer(
            l_summary,
            indices=-1,
            axis=1
        )

        summary_var = tf.Variable(
            initial_value=np.zeros((0, state_dim), dtype=np.float32),
            validate_shape=False,
            name="summary",
            trainable=False
        )

        self.l_embedding = l_embedding
        self.l_last_summary = l_last_summary
        self.l_summary = l_summary
        self.l_obs_input = obs_embedding_network.l_obs_input
        self.summary_var = summary_var
        self.output_dim = state_dim
        self.state_dim = state_dim
        self.embedding_dim = obs_embedding_network.embedding_dim
        self.input_vars = obs_embedding_network.input_vars
        self.output_layer = l_summary
        LayersPowered.__init__(self, [l_summary])

    def get_update_op(self, obs_var=None, action_var=None, **kwargs):
        summary = self.get_output(obs_var=obs_var, action_var=action_var, **kwargs)
        return tf.assign(self.summary_var, summary, validate_shape=False)

    def get_output(self, obs_var=None, action_var=None, **kwargs):
        if obs_var is None:
            obs_var = self.l_obs_input.input_var
        # we shift and pad the action into prev action, so that it aligns with the input of the action network
        # hence effectively, the last action is ignored
        return L.get_output(
            self.l_last_summary,
            {self.l_obs_input: obs_var},
            **kwargs
        )


class ActionNetwork(LayersPowered, Serializable):
    def __init__(self, env_spec, summary_network):
        Serializable.quick_init(self, locals())
        self.summary_network = summary_network
        l_summary_input = L.InputLayer(
            shape=(None, summary_network.state_dim),
        )
        action_dim = env_spec.action_space.flat_dim
        l_action = L.DenseLayer(
            l_summary_input,
            num_units=action_dim,
            nonlinearity=None
        )
        self.l_obs_input = summary_network.l_obs_input

        self.prev_state_var = tf.Variable(
            initial_value=np.zeros((0, summary_network.state_dim), dtype=np.float32),
            validate_shape=False,
            name="prev_state",
            trainable=False
        )
        # Just a placeholder
        self.l_summary_input = l_summary_input
        self.l_action = l_action
        self.output_layer = l_action
        self.state_dim = summary_network.state_dim
        self.action_dim = action_dim
        LayersPowered.__init__(self, [l_action, summary_network.output_layer])

    def get_partial_reset_op(self, dones_var):
        # upon reset: set corresponding entry to zero
        N = tf.shape(dones_var)[0]
        dones_var = tf.expand_dims(dones_var, 1)
        initial_prev_state = self.summary_network.summary_var

        return tf.group(
            tf.assign(
                self.prev_state_var,
                self.prev_state_var * (1. - dones_var) + initial_prev_state * dones_var,
                validate_shape=False
            )
        )

    def get_full_reset_op(self, dones_var):
        N = tf.shape(dones_var)[0]
        initial_prev_state = self.summary_network.summary_var

        return tf.group(
            tf.assign(
                self.prev_state_var,
                initial_prev_state,
                validate_shape=False
            )
        )

    def get_step_op(self, obs_var, **kwargs):
        flat_embedding_var = L.get_output(
            L.TemporalFlattenLayer(self.summary_network.l_embedding),
            {
                self.summary_network.l_obs_input: tf.expand_dims(obs_var, 1),
            }, **kwargs
        )

        l_step_input = L.InputLayer(
            shape=(None, self.summary_network.embedding_dim),
            input_var=flat_embedding_var,
        )
        l_step_prev_state = L.InputLayer(
            shape=(None, self.state_dim),
            input_var=self.prev_state_var
        )
        step_summary_var = L.get_output(
            self.summary_network.l_summary.get_step_layer(
                l_in=l_step_input,
                l_prev_hidden=l_step_prev_state,
            ), **kwargs
        )

        # import ipdb;
        # ipdb.set_trace()
        action_var = L.get_output(self.l_action, {self.l_summary_input: step_summary_var}, **kwargs)

        update_ops = [
            tf.assign(self.prev_state_var, step_summary_var)
        ]

        with tf.control_dependencies(update_ops):
            action_var = tf.identity(action_var)

        return action_var, dict()#prev_action=self.prev_action_var)

    @property
    def state_info_specs(self):
        return [
            # ("prev_action", self.action_dim),
        ]

    @property
    def state_info_keys(self):
        return [k for (k, _) in self.state_info_specs]

    def get_output(self, obs_var, summary_var, state_info_vars, **kwargs):

        action_summary_var = L.get_output(
            L.TemporalFlattenLayer(self.summary_network.l_summary),
            {
                self.summary_network.l_obs_input: obs_var,
                # self.summary_network.l_action_input: state_info_vars["prev_action"],
            },
            recurrent_state={
                self.summary_network.l_summary: summary_var,
            }
        )
        return L.get_output(
            L.TemporalUnflattenLayer(self.l_action, self.summary_network.l_obs_input),
            {self.l_summary_input: action_summary_var, self.summary_network.l_obs_input: obs_var},
            **kwargs
        )

    @property
    def recurrent(self):
        return True


class Net(object):
    def new_networks(self, env_spec):
        obs_embedding_net = StateObsEmbeddingNetwork(env_spec=env_spec)
        # action_embedding_net = ActionEmbeddingNetwork(env_spec=env_spec)
        summary_net = SummaryNetwork(
            env_spec=env_spec,
            obs_embedding_network=obs_embedding_net,
            # action_embedding_network=action_embedding_net
        )
        action_net = ActionNetwork(
            env_spec=env_spec,
            summary_network=summary_net,
        )
        return summary_net, action_net
