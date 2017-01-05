import numpy as np
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import GRUNetwork, MLP
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.rnn_utils import create_recurrent_network, NetworkType
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


class SoftmaxDefault(object):
    """
    Normal softmax. Parametrize via logits.
    """

    def __init__(self, dim):
        self.dim = dim

    @property
    def flat_dim(self):
        return self.dim

    def activate(self, x):
        return tf.nn.softmax(x)


class SoftmaxNormalized(object):
    """
    Normalized softmax. Normalize logits to unit standard deviation, and then apply temperature.
    """

    def __init__(self, dim, bias=1.0):
        self.dim = dim
        self.bias = bias

    @property
    def flat_dim(self):
        # The last dimension encodes the temperature (before an exponential transform)
        return self.dim + 1

    def activate(self, x):
        logits = x[:, :self.dim]
        temp = tf.exp(x[:, self.dim:] + self.bias)
        _, var_logits = tf.nn.moments(logits, axes=[1], keep_dims=True)
        std_logits = tf.sqrt(var_logits)
        norm_logits = logits / (std_logits + 1e-8)
        return tf.nn.softmax(norm_logits / temp)


class SoftmaxExactEntropy(object):
    """
    Directly parametrize the entropy of the softmax function.

    TODO
    """

    pass


class CategoricalRNNPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_dim=32,
            feature_network=None,
            state_include_action=True,
            hidden_nonlinearity=tf.tanh,
            network_type=NetworkType.GRU,
            weight_normalization=False,
            layer_normalization=False,
            action_param=None,
    ):
        Serializable.quick_init(self, locals())
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Discrete)
            super(CategoricalRNNPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            if state_include_action:
                input_dim = obs_dim + action_dim
            else:
                input_dim = obs_dim

            l_input = L.InputLayer(
                shape=(None, None, input_dim),
                name="input"
            )

            if feature_network is None:
                feature_dim = input_dim
                l_flat_feature = None
                l_feature = l_input
            else:
                feature_dim = feature_network.output_layer.output_shape[-1]
                l_flat_feature = feature_network.output_layer
                l_feature = L.OpLayer(
                    l_flat_feature,
                    extras=[l_input],
                    name="reshape_feature",
                    op=lambda flat_feature, input: tf.reshape(
                        flat_feature,
                        tf.pack([tf.shape(input)[0], tf.shape(input)[1], feature_dim])
                    ),
                    shape_op=lambda _, input_shape: (input_shape[0], input_shape[1], feature_dim)
                )

            if action_param is None:
                action_param = SoftmaxDefault(env_spec.action_space.n)

            self.action_param = action_param

            prob_network = create_recurrent_network(
                network_type,
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=self.action_param.flat_dim,  # env_spec.action_space.n,
                hidden_dim=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=self.action_param.activate,  # None,#tf.nn.softmax,
                weight_normalization=weight_normalization,
                layer_normalization=layer_normalization,
                name="prob_network",
            )

            self.prob_network = prob_network
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            flat_input_var = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name="flat_input")
            if feature_network is None:
                feature_var = flat_input_var
            else:
                feature_var = L.get_output(l_flat_feature, {feature_network.input_layer: flat_input_var})

            self.f_step_prob = tensor_utils.compile_function(
                [
                    flat_input_var,
                    prob_network.step_prev_state_layer.input_var
                ],
                L.get_output([
                    prob_network.step_output_layer,
                    prob_network.step_state_layer
                ], {prob_network.step_input_layer: feature_var})
            )

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.hidden_dim = hidden_dim
            self.state_dim = prob_network.state_dim

            self.prev_actions = None
            self.prev_states = None
            self.dist = RecurrentCategorical(env_spec.action_space.n)

            out_layers = [prob_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars, **kwargs):
        # n_batches = tf.shape(obs_var)[0]
        # n_steps = tf.shape(obs_var)[1]
        # obs_var = tf.reshape(obs_var, tf.pack([n_batches, n_steps, -1]))
        obs_var = tf.cast(obs_var, tf.float32)
        if self.state_include_action:
            prev_action_var = tf.cast(state_info_vars["prev_action"], tf.float32)
            all_input_var = tf.concat(2, [obs_var, prev_action_var])
        else:
            all_input_var = obs_var
        if self.feature_network is None:
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var},
                    **kwargs
                )
            )
        else:
            flat_input_var = tf.reshape(all_input_var, (-1, self.input_dim))
            return dict(
                prob=L.get_output(
                    self.prob_network.output_layer,
                    {self.l_input: all_input_var, self.feature_network.input_layer: flat_input_var},
                    **kwargs
                )
            )

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.prev_actions is None or len(dones) != len(self.prev_actions):
            self.prev_actions = np.zeros((len(dones), self.action_space.flat_dim))
            self.prev_states = np.zeros((len(dones), self.state_dim))

        if np.any(dones):
            self.prev_actions[dones] = 0.
            self.prev_states[dones] = self.prob_network.state_init_param.eval()  # get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.state_include_action:
            assert self.prev_actions is not None
            all_input = np.concatenate([
                flat_obs,
                self.prev_actions
            ], axis=-1)
        else:
            all_input = flat_obs
        probs, state_vec = self.f_step_prob(all_input, self.prev_states)
        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        prev_actions = self.prev_actions
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_states = state_vec
        agent_info = dict(prob=probs)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        return actions, agent_info

    @property
    @overrides
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        if self.state_include_action:
            return [
                ("prev_action", (self.action_dim,)),
            ]
        else:
            return []
