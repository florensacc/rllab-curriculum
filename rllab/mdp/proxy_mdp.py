from rllab.misc.overrides import overrides
from .base import MDP


class ProxyMDP(MDP):

    def __init__(self, mdp):
        self._mdp = mdp

    @overrides
    def reset(self):
        return self._mdp.reset()

    @property
    @overrides
    def action_dim(self):
        return self._mdp.action_dim

    @property
    def action_bounds(self):
        return self._mdp.action_bounds

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
    def step(self, action):
        return self._mdp.step(action)

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
        self._mdp.log_diagnostics(*args, **kwargs)

    def get_state(self):
        return self._mdp.get_state()

    def get_current_obs(self):
        return self._mdp.get_current_obs()

    @property
    def viewer(self):
        return self._mdp.viewer

    def action_from_key(self, key):
        return self._mdp.action_from_key(key)
