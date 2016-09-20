from rllab.core.serializable import Serializable
from sandbox.rocky.analogy.policies.base import AnalogyPolicy
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np

from sandbox.rocky.tf.policies import rnn_utils


class SingleRNNPolicy(AnalogyPolicy, LayersPowered, Serializable):
    """
    Use a single RNN for processing the demonstration as well as performing on the new env
    """

    def __init__(self, env_spec, name, rnn_hidden_size=32, rnn_hidden_nonlineairty=tf.nn.tanh,
                 mlp_hidden_nonlinearity=tf.nn.tanh, state_include_action=False, network_type=rnn_utils.NetworkType.GRU,
                 output_nonlinearity=None,
                 mlp_hidden_sizes=(32, 32),
                 layer_args=None):
        Serializable.quick_init(self, locals())
        with tf.variable_scope(name):
            AnalogyPolicy.__init__(self, env_spec=env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            # We have one more dimension to indicate whether the current input is the demonstration or the actual
            # environment
            if state_include_action:
                gru_input_dim = obs_dim + action_dim + 1
            else:
                gru_input_dim = obs_dim + 1

            embedding_network = rnn_utils.create_recurrent_network(
                network_type,
                input_shape=(gru_input_dim,),
                output_dim=rnn_hidden_size,
                hidden_dim=rnn_hidden_size,
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=None,
                name="prob_network"
            )

            l_summary = L.SliceLayer(summary_network.recurrent_layer, indices=-1, axis=1)

            self.summary_network = summary_network
            self.action_network = action_network
            self.l_summary = l_summary
            self.l_action_obs = l_obs
            self.l_summary_in = l_summary_in
            self.l_summary_input = summary_network.input_layer
            self.state_include_action = state_include_action
            self.summary_var = summary_var

            self.f_update_summary = tensor_utils.compile_function(
                [summary_network.input_var],
                tf.assign(summary_var, L.get_output(l_summary)),
            )

            self.f_action = tensor_utils.compile_function(
                [l_obs.input_var],
                L.get_output(action_network.output_layer),
            )

            self.gru_size = rnn_hidden_size

            LayersPowered.__init__(self, [summary_network.output_layer, action_network.output_layer])

    def action_sym(self, obs_var, state_info_vars):
        demo_obs_var = state_info_vars["demo_obs"]
        demo_action_var = state_info_vars["demo_action"]
        if self.state_include_action:
            summary_input_var = tf.concat(2, [demo_obs_var, demo_action_var])
        else:
            summary_input_var = demo_obs_var
        summary_var = L.get_output(self.l_summary, {self.summary_network.input_layer: summary_input_var})
        batch_size = tf.shape(obs_var)[0]
        n_steps = tf.shape(obs_var)[1]
        flat_obs_var = tf.reshape(obs_var, (-1, self.observation_space.flat_dim))
        flat_summary_var = tf.reshape(
            tf.tile(
                tf.expand_dims(summary_var, 1),
                tf.pack([1, n_steps, 1]),
            ),
            (-1, self.gru_size),
        )
        action_var = L.get_output(
            self.action_network.output_layer, {
                self.l_action_obs: flat_obs_var,
                self.l_summary_in: flat_summary_var
            }
        )

        return tf.reshape(action_var, tf.pack([batch_size, n_steps, self.action_space.flat_dim]))

    def apply_demo(self, path):
        demo_obs = path["observations"]
        demo_actions = path["actions"]
        if self.state_include_action:
            summary_input = np.concatenate([demo_obs, demo_actions], axis=1)
        else:
            summary_input = demo_obs
        self.f_update_summary([summary_input])

    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        action = self.f_action([flat_obs])
        return action[0], dict()
