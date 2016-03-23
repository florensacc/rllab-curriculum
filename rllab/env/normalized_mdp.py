import numpy as np

from rllab.core.serializable import Serializable
from rllab.env.proxy_mdp import ProxyMDP
from rllab.misc.overrides import overrides


def normalize(mdp):
    return NormalizedMDP(mdp)


class NormalizedMDP(ProxyMDP, Serializable):

    def __init__(self, mdp):
        super(NormalizedMDP, self).__init__(mdp)
        Serializable.quick_init(self, locals())

    @property
    @overrides
    def action_bounds(self):
        return -np.ones(self.action_dim), np.ones(self.action_dim)

    @overrides
    def step(self, action):
        lb, ub = self._mdp.action_bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return self._mdp.step(scaled_action)
