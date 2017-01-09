from rllab.misc.overrides import overrides
from rllab.misc.ext import AttrDict
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.exploration_strategies.base import ExplorationStrategy
import numpy as np
import numpy.random as nr


class GaussianStrategy(ExplorationStrategy, Serializable):
    """
    Naively applies Gaussian noise to actions
    """

    def __init__(self, env_spec, mu=0, sigma=0.3, **kwargs):
        assert isinstance(env_spec.action_space, Box)
        assert len(env_spec.action_space.shape) == 1
        Serializable.quick_init(self, locals())
        self.mu = mu
        self.sigma = sigma
        self.action_space = env_spec.action_space

    @overrides
    def reset(self):
        pass

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        action, _ = policy.get_action(observation)
        return self.get_modified_action(t, action)

    def get_modified_action(self, t, action):
        noise = self.mu + self.sigma * np.random.randn(len(action))
        # print("pure: ", action, "noise: ", noise)
        return np.clip(action + noise, self.action_space.low, self.action_space.high)
