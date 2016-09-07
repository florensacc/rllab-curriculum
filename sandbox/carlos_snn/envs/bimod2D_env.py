from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np

class BimodEnv(Env):
    def __init__(self,mu1=[-1,0],mu2=[1,0],sigma1=0.01,sigma2=0.01,rand_init=False):
        self.mu1=np.array(mu1)
        self.mu2=np.array(mu2)
        self.sigma1=sigma1 ##we suppose symetric Gaussians
        self.sigma2=sigma2 ##here sigma is in fact the variance, ie sigma**2.
        self.rand_init=rand_init
    @property
    def observation_space(self):
        return Box(low= -np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=5.0*np.linalg.norm(self.mu1), high=5.0*np.linalg.norm(self.mu1), shape=(2,))

    def reset(self):
        self._state = np.zeros(shape=(2,)) \
                    + int(self.rand_init)*((np.random.rand(2,)-0.5)*5*np.linalg.norm(self.mu1) ) ##mu1 is taken as largest
        observation = np.copy(self._state)
        return observation

    def reward_state(self,state):
        x = state
        mu=self.mu1
        n = 2     ##number of modes around 0
        A = np.array([[np.cos(2.*np.pi/n), -np.sin(2.*np.pi/n)], [np.sin(2.*np.pi/n), np.cos(2.*np.pi/n)]])  ##rotation matrix
        reward = -0.5 + 1./(2*np.sqrt(np.power(2.*np.pi,2.)*self.sigma1))*(np.exp(-0.5/self.sigma1*np.linalg.norm(x-mu)**2))
        for i in range(1,n):
            mu = np.dot(A,mu)
            reward += 1. / (2 * np.sqrt(np.power(2. * np.pi, 2.) * self.sigma1)) * (
                np.exp(-0.5 / self.sigma1 * np.linalg.norm(x - mu) ** 2))
        return reward
        # return float(- 0.5 + 1./(2.*np.sqrt(np.power(2.*np.pi,2.)*self.sigma1))*(np.exp(-0.5/self.sigma1*(np.linalg.norm(x)-self.mu1)**2)) )#\
    def step(self, action):
        self._state = self._state + action
        done = True
        next_observation = np.copy(self._state)
        reward =  self.reward_state(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)