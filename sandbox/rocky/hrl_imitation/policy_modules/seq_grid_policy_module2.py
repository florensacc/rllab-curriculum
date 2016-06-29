from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf

import sandbox.rocky.tf.core.layers as L
from rllab.core.serializable import Serializable
from rllab.envs.base import EnvSpec
from sandbox.rocky.hrl_imitation.low_policies.branching_categorical_mlp_policy1 import BranchingCategoricalMLPPolicy
from sandbox.rocky.tf.core.network import ConvMergeNetwork
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.product import Product


class SeqGridPolicyModule(object):
    """
    low-level policy receives partial observation. stochastic bottleneck
    """

    def new_high_policy(self, env_spec, subgoal_dim):
        subgoal_space = Discrete(subgoal_dim)
        return CategoricalMLPPolicy(
            name="high_policy",
            env_spec=EnvSpec(
                observation_space=env_spec.observation_space,
                action_space=subgoal_space,
            ),
            prob_network=ConvMergeNetwork(
                name="high_policy_network",
                input_shape=env_spec.observation_space.components[0].shape,
                extra_input_shape=(Product(env_spec.observation_space.components[1:]).flat_dim,),
                output_dim=subgoal_dim,
                hidden_sizes=(10, 10),
                conv_filters=(10, 10),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('SAME', 'SAME'),
                extra_hidden_sizes=tuple(),  # (10, 10),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.softmax,
            ),
        )

    def new_alt_high_policy(self, env_spec, subgoal_dim):
        subgoal_space = Discrete(subgoal_dim)
        return CategoricalMLPPolicy(
            name="alt_high_policy",
            env_spec=EnvSpec(
                observation_space=env_spec.observation_space,
                action_space=subgoal_space,
            ),
        )

    def new_low_policy(self, env_spec, subgoal_dim, bottleneck_dim):
        subgoal_space = Discrete(subgoal_dim)
        return BranchingCategoricalMLPPolicy(
            name="low_policy",
            env_spec=EnvSpec(
                observation_space=Product(env_spec.observation_space, subgoal_space),
                action_space=env_spec.action_space,
            ),
            shared_network=ConvMergeNetwork(
                name="low_policy_shared_network",
                input_shape=env_spec.observation_space.components[0].shape,
                extra_input_shape=(Product(env_spec.observation_space.components[1:]).flat_dim,),
                output_dim=32,  # env.action_space.flat_dim,
                hidden_sizes=tuple(),  # (10,),
                conv_filters=(10, 10),
                conv_filter_sizes=(3, 3),
                conv_strides=(1, 1),
                conv_pads=('SAME', 'SAME'),
                extra_hidden_sizes=tuple(),  # (10, 10),
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.tanh,
            ),
            subgoal_dim=subgoal_dim,
            hidden_sizes=(32,),
            hidden_nonlinearity=tf.nn.tanh,
            bottleneck_dim=bottleneck_dim,
        )
