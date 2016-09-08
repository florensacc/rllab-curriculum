from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step

import numpy as np


class StaticEnv(Env):
    def __init__(self, noise_level=0, start_state=0, *args, **kwargs):
        Env.__init__(self, *args, **kwargs)
        self.noise_level = noise_level
        self.start_state = start_state

    @property
    def observation_space(self):
        return Box(low=self.start_state, high=self.start_state, shape=(1,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(1,))

    def reset(self):
        return self.start_state

    def step(self, action):
        assert action.shape == (1,)
        action = action[0]
        noise = np.random.normal()
        reward = -action**2 - self.noise_level * noise**2
        return Step(observation=self.start_state, reward=reward, done=False, noise=[noise])

    def render(self):
        print('current state:', self._state)

    def get_state(self):
        return self.start_state

    def set_state_tmp(self, state, restore=True):
        yield
        pass
