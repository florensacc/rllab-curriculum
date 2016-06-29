from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import GRUNetwork, MLP, ConvMergeNetwork
from sandbox.rocky.tf.spaces.product import Product
import tensorflow as tf
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.distributions.categorical import Categorical


class ApproximatePosterior(LayersPowered):
    """
    The approximate posterior takes in the sequence of states and actions, and predicts the hidden state based on the
    sequence. It is structured as a GRU.
    """

    def __init__(self, env_spec, subgoal_dim, subgoal_interval, feature_dim=20, hidden_dim=10):
        """
        :type env_spec: EnvSpec
        :param env_spec:
        :return:
        """
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        l_in = L.InputLayer(
            name="input",
            shape=(None, None, obs_dim + action_dim),
        )
        l_obs = L.SliceLayer(
            l_in,
            indices=slice(obs_dim),
            axis=-1,
            name="obs_input"
        )
        l_action = L.SliceLayer(
            l_in,
            indices=slice(obs_dim, obs_dim + action_dim),
            axis=-1,
            name="action_input"
        )

        feature_network = ConvMergeNetwork(
            name="feature_network",
            input_layer=L.reshape(l_obs, (-1, env_spec.observation_space.flat_dim), name="reshape_obs"),
            input_shape=env_spec.observation_space.components[0].shape,
            extra_input_shape=(Product(env_spec.observation_space.components[1:]).flat_dim,),
            output_dim=feature_dim,
            hidden_sizes=tuple(),
            conv_filters=(10, 10),
            conv_filter_sizes=(3, 3),
            conv_strides=(1, 1),
            conv_pads=('SAME', 'SAME'),
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
        )
        l_reshaped_feature = L.reshape(
            feature_network.output_layer,
            shape=(-1, subgoal_interval, feature_dim),
            name="reshaped_feature"
        )

        l_preprocess_in = L.reshape(
            L.concat([l_reshaped_feature, l_action], name="preprocess_in", axis=2),
            (-1, feature_dim + action_dim),
            name="preprocess_in_flat",
        )
        preprocess_network = MLP(
            name="preprocess_network",
            input_shape=(feature_dim + action_dim,),
            output_dim=feature_dim,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.tanh,
            input_layer=l_preprocess_in,
            hidden_sizes=(feature_dim,),
        )

        subgoal_input_layer = L.reshape(
            preprocess_network.output_layer,
            (-1, subgoal_interval, feature_dim),
            name="subgoal_in_reshape"
        )

        subgoal_network = GRUNetwork(
            name="h_network",
            input_shape=(subgoal_input_layer.output_shape[-1],),
            output_dim=subgoal_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=tf.nn.tanh,
            output_nonlinearity=tf.nn.softmax,
            input_layer=subgoal_input_layer,
        )
        l_subgoal_probs = L.SliceLayer(
            subgoal_network.output_layer,
            indices=subgoal_interval - 1,
            axis=1,
            name="subgoal_probs"
        )
        self.subgoal_dim = subgoal_dim
        self.l_in = l_in
        self.l_obs = l_obs
        self.l_action = l_action
        self.l_subgoal_probs = l_subgoal_probs
        LayersPowered.__init__(self, [l_subgoal_probs])

    def dist_info_sym(self, obs_var, action_var):
        assert obs_var.get_shape().ndims == 3
        assert action_var.get_shape().ndims == 3
        action_var = tf.cast(action_var, tf.float32)
        prob = L.get_output(self.l_subgoal_probs, {self.l_obs: obs_var, self.l_action: action_var})
        return dict(prob=prob)

    @property
    def distribution(self):
        return Categorical(self.subgoal_dim)
