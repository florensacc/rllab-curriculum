from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np


class MultiMod2DEnv(Env):
    """
    This is a single time-step MDP where the action taken corresponds to the next state (in a 2D plane).
    The reward has a multi-modal gaussian shape, with the mode means set in a circle around the origin.
    """
    def __init__(self, mu=(1, 0), sigma=0.01, n=2, rand_init=False):
        self.mu = np.array(mu)
        self.sigma = sigma  #we suppose symetric Gaussians
        self.n = n
        self.rand_init = rand_init

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=5.0 * np.linalg.norm(self.mu), high=5.0 * np.linalg.norm(self.mu), shape=(2,))

    def reset(self):
        self._state = np.zeros(shape=(2,)) \
                      + int(self.rand_init) * (
                        (np.random.rand(2, ) - 0.5) * 5 * np.linalg.norm(self.mu) )  ##mu is taken as largest
        observation = np.copy(self._state)
        return observation

    def reward_state(self, state):
        x = state
        mu = self.mu
        A = np.array([[np.cos(2. * np.pi / self.n), -np.sin(2. * np.pi / self.n)],
                      [np.sin(2. * np.pi / self.n), np.cos(2. * np.pi / self.n)]])  ##rotation matrix
        reward = -0.5 + 1. / (2 * np.sqrt(np.power(2. * np.pi, 2.) * self.sigma)) * (
        np.exp(-0.5 / self.sigma * np.linalg.norm(x - mu) ** 2))
        for i in range(1, self.n):
            mu = np.dot(A, mu)
            reward += 1. / (2 * np.sqrt(np.power(2. * np.pi, 2.) * self.sigma)) * (
                np.exp(-0.5 / self.sigma * np.linalg.norm(x - mu) ** 2))
        return reward

    def step(self, action):
        self._state += action
        done = True
        next_observation = np.copy(self._state)
        reward = self.reward_state(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)

    def log_diagnostics(self, paths):
        # to count the modes I need the current policy!
        pass
