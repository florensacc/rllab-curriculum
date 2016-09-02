from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


class PointEnv(Env):
    def __init__(self, state_dim=2, end_early=True, *args, **kwargs):
        Env.__init__(self, *args, **kwargs)
        self.state_dim = state_dim
        self.end_early = end_early

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(self.state_dim,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(self.state_dim,))

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(self.state_dim,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        reward = -np.linalg.norm(self._state)
        done = self.end_early and np.all(self._state < 0.01)
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print 'current state:', self._state

    def sim_vectorized(self, state, actions):
        """
        Takes an array of shape (num_rollouts, action_dim=2) and returns a dict containing results.
        """
        observations = state + actions
        rewards = -np.sum(observations**2, axis=1)**0.5
        dones = np.all(np.abs(observations) < 0.01, axis=1)
        return dict(
            observations=observations,
            rewards=rewards,
            dones=dones,
            final_states=observations,
        )

    def get_state(self):
        return self._state

    def set_state_tmp(self, state, restore=True):
        if restore:
            prev_state = self._state
        self._state = state
        yield
        if restore:
            self._state = prev_state
