from .base import MDP

class ProxyMDP(MDP):

    def __init__(self, base_mdp):
        self._base_mdp = base_mdp

    def sample_initial_states(self, n):
        return self._base_mdp.sample_initial_states(n)

    @property
    def action_set(self):
        return self._base_mdp.action_set

    @property
    def action_dims(self):
        return self._base_mdp.action_dims

    @property
    def observation_shape(self):
        return self._base_mdp.observation_shape

    def step(self, states, action_indices):
        return self._base_mdp.step(state, action_indices)
