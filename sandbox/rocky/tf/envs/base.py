from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box as TheanoBox
from rllab.spaces.discrete import Discrete as TheanoDiscrete
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box


def to_tf_space(space):
    if isinstance(space, TheanoBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, TheanoDiscrete):
        return Discrete(space.n)
    else:
        raise NotImplementedError


class TfEnv(ProxyEnv):
    @property
    def observation_space(self):
        return to_tf_space(self.wrapped_env.observation_space)

    @property
    def action_space(self):
        return to_tf_space(self.wrapped_env.action_space)
