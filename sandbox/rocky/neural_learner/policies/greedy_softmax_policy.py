from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.parameterized import Parameterized
import numpy as np


class GreedySoftmaxPolicy(Parameterized, Serializable):

    def __init__(self, wrapped_policy):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        self.wrapped_policy = wrapped_policy

    def get_params_internal(self, **tags):
        return self.wrapped_policy.get_params(**tags)

    @property
    def vectorized(self):
        return self.wrapped_policy.vectorized

    def get_action(self, obs):
        _, agent_info = self.wrapped_policy.get_action(obs)
        action = np.argmax(agent_info['prob'], axis=-1)
        return action, agent_info

    def get_actions(self, obs):
        _, agent_info = self.wrapped_policy.get_actions(obs)
        actions = np.argmax(agent_info['prob'], axis=-1)
        return actions, agent_info

    def reset(self, dones):
        self.wrapped_policy.reset(dones)
