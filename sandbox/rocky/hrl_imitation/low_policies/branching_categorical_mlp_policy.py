from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.spaces.box import Box
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np
from rllab.misc import special


class BranchingCategoricalMLPPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            shared_network,
            subgoal_dim,
            bottleneck_dim,
            bottleneck_std_threshold=1e-3,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
    ):
        """
        :param env_spec: A spec for the mdp.
        :param hidden_sizes: list of sizes for the fully connected hidden layers
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param prob_network: manually specified network for this policy, other network params
        are ignored
        :return:
        """
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        with tf.variable_scope(name):
            l_last = shared_network.output_layer
            # map to mean and std of bottleneck
            l_bottleneck_mean = L.DenseLayer(
                l_last,
                num_units=bottleneck_dim,
                nonlinearity=tf.nn.tanh,
                name="bottleneck_mean"
            )
            l_bottleneck_std = L.DenseLayer(
                l_last,
                num_units=bottleneck_dim,
                nonlinearity=tf.exp,
                name="bottleneck_std"
            )
            l_bottleneck_std = L.OpLayer(
                l_bottleneck_std,
                op=lambda x: tf.minimum(tf.maximum(x, bottleneck_std_threshold), bottleneck_std_threshold),
                shape_op=lambda x: x,
                name="bottleneck_std_clipped",
            )
            l_bottleneck_epsilon = L.InputLayer(shape=(None, bottleneck_dim), name="l_bottleneck_epsilon")

            l_bottleneck = L.OpLayer(
                l_bottleneck_mean, extras=[l_bottleneck_std, l_bottleneck_epsilon],
                op=lambda mean, std, epsilon: mean + std * epsilon,
                shape_op=lambda x, *args: x,
                name="bottleneck"
            )

            prob_networks = []

            for subgoal in xrange(subgoal_dim):
                prob_network = MLP(
                    input_layer=l_bottleneck,
                    output_dim=env_spec.action_space.n,
                    hidden_sizes=hidden_sizes,
                    hidden_nonlinearity=hidden_nonlinearity,
                    output_nonlinearity=tf.nn.softmax,
                    name="prob_network_%d" % subgoal,
                )
                prob_networks.append(prob_network)

            self.prob_networks = prob_networks
            self.l_probs = [net.output_layer for net in prob_networks]
            self.l_obs = [x for x in L.get_all_layers(shared_network.input_layer) if isinstance(x, L.InputLayer)][0]
            self.l_bottleneck_mean = l_bottleneck_mean
            self.l_bottleneck_std = l_bottleneck_std
            self.l_bottleneck_epsilon = l_bottleneck_epsilon
            self.bottleneck_dim = bottleneck_dim

            self.bottleneck_dist = bottleneck_dist = DiagonalGaussian(dim=bottleneck_dim)
            self.subgoal_dim = subgoal_dim
            self.dist = Categorical(env_spec.action_space.n)
            self.shared_network = shared_network

            self.bottleneck_space = bottleneck_space = Box(low=-1, high=1, shape=(bottleneck_dim,))

            super(BranchingCategoricalMLPPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [net.output_layer for net in prob_networks])

            obs_var = self.observation_space.new_tensor_variable(
                "obs",
                extra_dims=1,
            )
            epsilon_var = bottleneck_space.new_tensor_variable(
                "bottleneck_epsilon",
                extra_dims=1,
            )

            self.f_prob = tensor_utils.compile_function(
                [obs_var, epsilon_var],
                self.dist_info_sym(obs_var, dict(bottleneck_epsilon=epsilon_var))["prob"],
            )
            self.f_bottleneck_dist_info = tensor_utils.compile_function(
                [obs_var],
                self.bottleneck_dist_info_sym(obs_var)
            )

    def bottleneck_dist_info_sym(self, obs_var):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        subgoals = obs_var[:, self.observation_space.flat_dim - self.subgoal_dim:]
        mean, std = L.get_output(
            [self.l_bottleneck_mean, self.l_bottleneck_std],
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
            }
        )
        return dict(mean=mean, log_std=tf.log(std))

    def dist_info_sym(self, obs_var, state_info_vars):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        subgoals = obs_var[:, self.observation_space.flat_dim - self.subgoal_dim:]
        bottleneck_epsilon = state_info_vars["bottleneck_epsilon"]
        prob_vars = L.get_output(
            self.l_probs,
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
                self.l_bottleneck_epsilon: bottleneck_epsilon,
            }
        )
        probs = tf.batch_matmul(
            tf.expand_dims(subgoals, 1),
            tf.transpose(tf.pack(prob_vars), (1, 0, 2))
        )[:, 0, :]
        return dict(prob=probs)

    def dist_info(self, obs, state_infos):
        return dict(prob=self.f_prob(obs, state_infos["bottleneck_epsilon"]))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_action(self, observation):
        bottleneck_epsilon = np.random.normal(size=(self.bottleneck_dim,))
        flat_obs = self.observation_space.flatten(observation)
        dist_info = self.dist_info([flat_obs], dict(bottleneck_epsilon=[bottleneck_epsilon]))
        act = special.weighted_sample(dist_info["prob"], range(self.action_space.n))
        return act, dist_info

    def get_actions(self, observations):
        N = len(observations)
        bottleneck_epsilon = np.random.normal(size=(N, self.bottleneck_dim))
        flat_obses = self.observation_space.flatten_n(observations)
        dist_info = self.dist_info(flat_obses, dict(bottleneck_epsilon=bottleneck_epsilon))
        act = [special.weighted_sample(p, range(self.action_space.n)) for p in dist_info["prob"]]
        return act, dist_info

    @property
    def distribution(self):
        return self.dist
