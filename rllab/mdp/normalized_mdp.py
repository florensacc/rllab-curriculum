import numpy as np
from rllab.mdp.base import MDP, ControlMDP, SymbolicMDP
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


def normalize(mdp):
    if isinstance(mdp, SymbolicMDP):
        return NormalizedSymbolicMDP(mdp)
    elif isinstance(mdp, ControlMDP):
        return NormalizedControlMDP(mdp)
    else:
        return NormalizedMDP(mdp)


class NormalizedMDP(MDP, Serializable):

    def __init__(self, mdp):
        self._mdp = mdp
        Serializable.__init__(self, mdp)

    @overrides
    def reset(self):
        return self._mdp.reset()

    @property
    @overrides
    def action_dim(self):
        return self._mdp.action_dim

    @property
    @overrides
    def action_dtype(self):
        return self._mdp.action_dtype

    @property
    @overrides
    def observation_dtype(self):
        return self._mdp.observation_dtype

    @property
    @overrides
    def observation_shape(self):
        return self._mdp.observation_shape

    @overrides
    def step(self, state, action):
        return self._mdp.step(state, action)

    @overrides
    def start_viewer(self):
        return self._mdp.start_viewer()

    @overrides
    def stop_viewer(self):
        return self._mdp.stop_viewer()

    @overrides
    def plot(self, *args, **kwargs):
        return self._mdp.plot(*args, **kwargs)

    @overrides
    def log_extra(self, *args, **kwargs):
        self._mdp.log_extra(*args, **kwargs)


class NormalizedControlMDP(NormalizedMDP, ControlMDP, Serializable):

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
        return self._mdp.step(state, scaled_action)


class NormalizedSymbolicMDP(NormalizedControlMDP, SymbolicMDP, Serializable):

    def __init__(self, mdp):
        super(NormalizedSymbolicMDP, self).__init__(mdp)
        SymbolicMDP.__init__(self)
        Serializable.__init__(self, mdp)

    @property
    def state_shape(self):
        return self._mdp.state_shape
