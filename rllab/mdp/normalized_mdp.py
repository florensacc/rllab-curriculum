import numpy as np
from rllab.mdp.base import MDP, ControlMDP, SymbolicMDP
from rllab.core.serializable import Serializable
from rllab.mdp.proxy_mdp import ProxyMDP
from rllab.misc.overrides import overrides


def normalize(mdp):
    if isinstance(mdp, SymbolicMDP):
        return NormalizedSymbolicMDP(mdp)
    elif isinstance(mdp, ControlMDP):
        return NormalizedControlMDP(mdp)
    else:
        return ProxyMDP(mdp)


class NormalizedControlMDP(ProxyMDP, ControlMDP, Serializable):

    def __init__(self, mdp):
        super(NormalizedControlMDP, self).__init__(mdp)
        ControlMDP.__init__(self)
        Serializable.__init__(self, mdp)

    @property
    @overrides
    def action_bounds(self):
        return -np.ones(self.action_dim), np.ones(self.action_dim)

    @overrides
    def step(self, state, action):
        lb, ub = self._mdp.action_bounds
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return self._mdp.step(state, scaled_action)


class NormalizedSymbolicMDP(ProxyMDP, SymbolicMDP, Serializable):

    def __init__(self, mdp):
        super(NormalizedSymbolicMDP, self).__init__(mdp)
        SymbolicMDP.__init__(self)
        Serializable.__init__(self, mdp)

    @property
    def state_shape(self):
        return self._mdp.state_shape

