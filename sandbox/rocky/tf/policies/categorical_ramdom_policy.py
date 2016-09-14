from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces.discrete import Discrete

from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import logger
from rllab.misc.overrides import overrides
import tensorflow as tf
import numpy as np

"""
A policy that always samples uniformly from available actions.
This is a baseline policy.
"""
class CategoricalRandomPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
    ):
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)
        self._env_spec = env_spec
        self.n_actions = env_spec.action_space.n
        self.prob = np.ones(self.n_actions,dtype=np.float32) * 1./self.n_actions
        self._dist = Categorical(self.n_actions)

        prob_network = ConvNetwork(
            input_shape=env_spec.observation_space.shape,
            output_dim=env_spec.action_space.n,
            conv_filters=[],
            conv_filter_sizes=[],
            conv_strides=[],
            conv_pads=[],
            hidden_sizes=[],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.softmax,
            name="prob_network",
        )


        super(CategoricalRandomPolicy, self).__init__(env_spec)
        LayersPowered.__init__(self, [prob_network.output_layer])

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        return dict(prob=self.prob)

    @overrides
    def dist_info(self, obs, state_infos=None):
        return dict(prob=self.prob)

    @overrides
    def get_action(self, observation):
        prob = self.prob
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        probs = np.asarray([
            self.prob for i in range(len(observations))
        ])
        actions = list(map(self.action_space.weighted_sample, probs))
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist
