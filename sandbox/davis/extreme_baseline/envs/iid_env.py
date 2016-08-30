from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


class IIDEnv(Env):
    def __init__(self, state_dim=2, *args, **kwargs):
        Env.__init__(self, *args, **kwargs)
        self.state_dim = state_dim

    @property
    def observation_space(self):
        return Box(low=-1, high=1, shape=(self.state_dim,))

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(self.state_dim,))

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(self.state_dim,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        reward = -np.linalg.norm(self._state - action)**2
        next_observation = self.reset()
        return Step(observation=next_observation, reward=reward, done=False, noise=next_observation)

    def render(self):
        print 'current state:', self._state

    def get_state(self):
        return self._state

    def set_state_tmp(self, state, restore=True):
        if restore:
            prev_state = self._state
        self._state = state
        yield
        if restore:
            self._state = prev_state
