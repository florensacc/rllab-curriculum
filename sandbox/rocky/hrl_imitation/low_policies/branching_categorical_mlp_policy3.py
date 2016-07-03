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

            l_bottleneck = L.DenseLayer(
                l_last,
                num_units=bottleneck_dim,
                nonlinearity=tf.nn.tanh,
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
            self.l_bottleneck = l_bottleneck
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
                self.dist_info_sym(obs_var)["prob"],
            )

    def bottleneck_sym(self, obs_var):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        return L.get_output(
            self.l_bottleneck,
            {self.l_obs: high_obs}
        )

    def get_all_probs(self, obs_var, state_info_vars=None):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        prob_vars = L.get_output(
            self.l_probs,
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
            }
        )
        return prob_vars

    def get_subgoal_probs(self, all_probs, subgoals):
        return tf.batch_matmul(
            tf.expand_dims(subgoals, 1),
            tf.transpose(tf.pack(all_probs), (1, 0, 2))
        )[:, 0, :]

    def dist_info_sym(self, obs_var, state_info_vars=None):
        high_obs = obs_var[:, :self.observation_space.flat_dim - self.subgoal_dim]
        subgoals = obs_var[:, self.observation_space.flat_dim - self.subgoal_dim:]
        prob_vars = L.get_output(
            self.l_probs,
            {
                self.l_obs: tf.cast(high_obs, tf.float32),
            }
        )
        probs = tf.batch_matmul(
            tf.expand_dims(subgoals, 1),
            tf.transpose(tf.pack(prob_vars), (1, 0, 2))
        )[:, 0, :]
        return dict(prob=probs)

    def dist_info(self, obs, state_infos=None):
        return dict(prob=self.f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        dist_info = self.dist_info([flat_obs])
        act = special.weighted_sample(dist_info["prob"], range(self.action_space.n))
        return act, dist_info

    def get_actions(self, observations):
        N = len(observations)
        flat_obses = self.observation_space.flatten_n(observations)
        dist_info = self.dist_info(flat_obses)
        act = [special.weighted_sample(p, range(self.action_space.n)) for p in dist_info["prob"]]
        return act, dist_info

    @property
    def distribution(self):
        return self.dist
