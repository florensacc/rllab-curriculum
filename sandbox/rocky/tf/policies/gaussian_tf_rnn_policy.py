import numpy as np
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf

from rllab.spaces import Box
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.distributions.recurrent_diagonal_gaussian import RecurrentDiagonalGaussian
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.rnn_utils import create_recurrent_network, NetworkType
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger


class DistHead(object):
    """
    Represent a specific parametrization of the probability density function.
    """

    head_dim = None

    def dist_info_sym(self, head_var):
        raise NotImplementedError

    @property
    def params(self):
        return []


class GlobalDiagonalGaussian(DistHead):
    def __init__(self, action_dim, min_std=1e-4):
        self.action_dim = action_dim
        self.log_std = tf.Variable(initial_value=np.zeros(action_dim), dtype=tf.float32, name="log_std")
        self.min_std = min_std

    @property
    def head_dim(self):
        return self.action_dim

    @property
    def params(self):
        return [self.log_std]

    def dist_info_sym(self, head_var):
        # add the min std
        if self.min_std > 0:
            log_min_std = np.asarray([np.log(self.min_std)] * self.action_dim, dtype=np.float32)
            bounded_log_std = tf.reduce_logsumexp(tf.pack([self.log_std, log_min_std]), reduction_indices=[0])
        else:
            bounded_log_std = self.log_std
        N = tf.shape(head_var)[0]
        T = tf.shape(head_var)[1]
        stacked_log_std = tf.tile(tf.reshape(bounded_log_std, (1, 1, -1)), tf.pack([N, T, 1]))
        return dict(
            mean=head_var,
            log_std=stacked_log_std,
        )


class DependentDiagonalGaussian(DistHead):
    def __init__(self, action_dim, min_std=1e-4):
        self.action_dim = action_dim
        self.min_std = min_std

    @property
    def head_dim(self):
        return self.action_dim * 2

    @property
    def params(self):
        return []

    def dist_info_sym(self, head_var):
        # add the min std
        mean, log_std = tf.split(split_dim=2, num_split=2, value=head_var)
        N = tf.shape(head_var)[0]
        T = tf.shape(head_var)[1]
        if self.min_std > 0:
            log_min_std = tf.tile(
                np.asarray(np.reshape(np.log(self.min_std), (1, 1, 1)), dtype=np.float32),
                tf.pack([N, T, self.action_dim])
            )
            bounded_log_std = tf.reduce_logsumexp(tf.pack([log_std, log_min_std]), reduction_indices=[0])
        else:
            bounded_log_std = log_std
        return dict(
            mean=mean,
            log_std=bounded_log_std,
        )


class GaussianTfRNNPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            cell,
            feature_network=None,
            state_include_action=True,
            dist_head_cls=GlobalDiagonalGaussian,
    ):
        Serializable.quick_init(self, locals())
        """
        :param env_spec: A spec for the env.
        """
        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Box)
            super(GaussianTfRNNPolicy, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            dist_head = dist_head_cls(action_dim)

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

            head_network = create_recurrent_network(
                NetworkType.TF_RNN,
                input_shape=(feature_dim,),
                input_layer=l_feature,
                output_dim=dist_head.head_dim,
                output_nonlinearity=None,
                cell=cell,
                name="head_network",
            )

            self.head_network = head_network
            self.dist_head = dist_head
            self.feature_network = feature_network
            self.l_input = l_input
            self.state_include_action = state_include_action

            self.input_dim = input_dim
            self.action_dim = action_dim
            self.state_dim = head_network.state_dim

            self.prev_actions = None
            self.prev_states = None
            self.deterministic = False
            self.dist = RecurrentDiagonalGaussian(action_dim)

            out_layers = [head_network.output_layer]
            if feature_network is not None:
                out_layers.append(feature_network.output_layer)

            LayersPowered.__init__(self, out_layers)

            obs_var = env_spec.observation_space.new_tensor_variable("obs", extra_dims=1)
            state_info_vars = [
                tf.placeholder(dtype=tf.float32, shape=(None,) + shape, name=k)
                for k, shape in self.state_info_specs
                ]
            prev_state_var = tf.placeholder(dtype=tf.float32, shape=(None, head_network.state_dim), name="prev_state")

            recurrent_state_output = dict()

            dist_info_vars = self.dist_info_sym(
                tf.expand_dims(obs_var, 1),
                dict(zip(self.state_info_keys, [tf.expand_dims(x, 1) for x in state_info_vars])),
                recurrent_state={head_network.recurrent_layer: prev_state_var},
                recurrent_state_output=recurrent_state_output
            )

            final_state_var = recurrent_state_output[head_network.recurrent_layer]

            self.f_step = tensor_utils.compile_function(
                inputs=[obs_var] + state_info_vars + [prev_state_var],
                outputs=[dist_info_vars["mean"][:, 0, :], dist_info_vars["log_std"][:, 0, :], final_state_var],
            )

    def get_params_internal(self, **tags):
        return LayersPowered.get_params_internal(self, **tags) + self.dist_head.params

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars, **kwargs):
        obs_var = tf.cast(obs_var, tf.float32)
        if self.state_include_action:
            prev_action_var = tf.cast(state_info_vars["prev_action"], tf.float32)
            all_input_var = tf.concat(2, [obs_var, prev_action_var])
        else:
            all_input_var = obs_var
        if self.feature_network is None:
            head = L.get_output(
                self.head_network.output_layer,
                {self.l_input: all_input_var},
                **kwargs
            )
        else:
            flat_input_var = tf.reshape(all_input_var, (-1, self.input_dim))
            head = L.get_output(
                self.head_network.output_layer,
                {self.l_input: all_input_var, self.feature_network.input_layer: flat_input_var},
                **kwargs
            )
        return self.dist_head.dist_info_sym(head)

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
            self.prev_states[dones] = self.head_network.state_init_param.eval()  # get_value()

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def configure(self, **kwargs):
        if 'deterministic' in kwargs:
            self.deterministic = kwargs['deterministic']

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
        action_means, log_stds, state_vec = self.f_step(all_input, self.prev_states)
        if self.deterministic:
            actions = action_means
        else:
            actions = np.random.normal(size=action_means.shape) * np.exp(log_stds) + action_means
        prev_actions = self.prev_actions
        agent_info = dict(mean=action_means, log_std=log_stds)
        if self.state_include_action:
            agent_info["prev_action"] = np.copy(prev_actions)
        self.prev_actions = self.action_space.flatten_n(actions)
        self.prev_states = state_vec
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

    def log_diagnostics(self, paths):
        log_stds = np.vstack([path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))
