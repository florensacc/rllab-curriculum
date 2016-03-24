import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides


def normalize(mdp):
    return NormalizedEnv(mdp)


class NormalizedEnv(ProxyEnv, Serializable):

    def __init__(self, mdp):
        super(NormalizedEnv, self).__init__(mdp)
        Serializable.quick_init(self, locals())

    @property
    @overrides
    def action_space(self):
        ub = np.ones(self._wrapped_env.action_space.shape)
        return spaces.Box(-1*ub, ub)

    @overrides
    def step(self, action):
        lb, ub = self._wrapped_env.action_bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return self._wrapped_env.step(scaled_action)

