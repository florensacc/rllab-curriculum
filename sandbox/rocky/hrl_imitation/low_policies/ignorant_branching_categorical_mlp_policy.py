

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
from sandbox.rocky.hrl_imitation.low_policies.branching_categorical_mlp_policy import BranchingCategoricalMLPPolicy


class IgnorantBranchingCategoricalMLPPolicy(BranchingCategoricalMLPPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            # shared_network,
            subgoal_dim,
            bottleneck_dim,
            bottleneck_std_threshold=1e-10,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.tanh,
    ):
        Serializable.quick_init(self, locals())
        l_in = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim), name="input")
        slice_start = env_spec.observation_space.components[0].components[0].flat_dim
        slice_end = env_spec.observation_space.components[0].flat_dim

        l_sliced_in = L.SliceLayer(l_in, name="sliced_input", indices=slice(slice_start, slice_end), axis=-1)

        shared_network = MLP(
            input_shape=(slice_end-slice_start,),
            input_layer=l_sliced_in,
            hidden_sizes=tuple(),
            hidden_nonlinearity=None,
            output_dim=slice_end-slice_start,
            output_nonlinearity=None,
            name="dummy_shared",
        )
        BranchingCategoricalMLPPolicy.__init__(self, name=name, env_spec=env_spec, subgoal_dim=subgoal_dim,
                                               bottleneck_dim=bottleneck_dim, shared_network=shared_network,
                                               bottleneck_std_threshold=bottleneck_std_threshold,
                                               hidden_sizes=hidden_sizes, hidden_nonlinearity=hidden_nonlinearity)
