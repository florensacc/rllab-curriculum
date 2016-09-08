


from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import random


class BimodalPointEnv(Env):

    def __init__(self):
        self.target_point = None
        self.reset_trial()

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset_trial(self):
        self.target_point = np.array(random.choice([
            [1, 1],
            [-1, -1],
        ]))
        return self.reset()

    def reset(self):
        self._state = np.array([0, 0])
        return np.copy(self._state)

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        tx, ty = self.target_point
        reward = - np.sum(np.square(self._state - self.target_point)) ** 0.5
        done = abs(x - tx) < 0.01 and abs(y - ty) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)
