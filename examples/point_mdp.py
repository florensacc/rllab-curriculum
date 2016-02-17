from rllab.mdp.base import MDP
import numpy as np


class PointMDP(MDP):

    @property
    def observation_shape(self):
        return (2,)

    @property
    def action_dim(self):
        return 2

    @property
    def observation_dtype(self):
        return 'float32'

    @property
    def action_dtype(self):
        return 'float32'

    @property
    def action_bounds(self):
        return - 0.1 * np.ones(2), 0.1 * np.ones(2)

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = - (x**2 - y**2) ** 0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return next_observation, reward, done

    def plot(self):
        print 'current state:', self._state
