from .proxy_mdp import ProxyMDP

class ObsTransformer(ProxyMDP):

    def __init__(self, base_mdp, obs_transform):
        super(ObsTransformer, self).__init__(base_mdp)
        self._obs_transform = obs_transform
        _, obs = base_mdp.sample_initial_state()
        self._observation_shape = obs_transform(obs).shape

    def sample_initial_states(self, n):
        states, obs = self._base_mdp.sample_initial_states(n)
        return states, map(self._obs_transform, obs)

    @property
    def observation_shape(self):
        return self._observation_shape

    def step(self, states, action_indices):
        next_states, obs, rewards, dones, effective_steps = self._base_mdp.step(states, action_indices)
        return next_states, map(self._obs_transform, obs), rewards, dones, effective_steps
