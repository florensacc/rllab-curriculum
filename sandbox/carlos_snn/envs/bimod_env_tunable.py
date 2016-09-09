from rllab.envs.base import Env
from rllab.envs.base import Step
from rllab.spaces import Box
import numpy as np


class BimodEnv(Env):
    def __init__(self, eps=0, disp=0, mu1=-1, mu2=1, sigma1=0.01, sigma2=0.01, rand_init=False):
        self.eps = eps
        self.mu1 = mu1-disp
        self.mu2 = mu2-disp
        self.sigma1 = sigma1  ##here sigma is in fact the variance, ie sigma**2.
        self.sigma2 = sigma2
        self.rand_init = rand_init

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(1,))

    @property
    def action_space(self):
        return Box(low=5.0 * self.mu1, high=5.0 * self.mu2, shape=(1,))

    def reset(self):
        self._state = np.zeros(shape=(1,)) + int(self.rand_init)*(
                    np.random.rand(1,)*5*(self.mu2-self.mu1)+self.mu1)
        observation = np.copy(self._state)
        return observation

    def reward_state(self,state):
        x, = state
        return - 0.5 + (0.5+self.eps)*1./(np.sqrt(2.*np.pi*self.sigma1))*(np.exp(-0.5/self.sigma1*(x-self.mu1)**2)) \
                     + (0.5-self.eps)*1./(np.sqrt(2.*np.pi*self.sigma2))*(np.exp(-0.5/self.sigma2*(x-self.mu2)**2))

    def step(self, action):
        # print 'before taking action {}, state= {}'.format(action, self._state)
        self._state = self._state + action
        # print 'after step state= ', self._state
        x, = self._state
        done = True
        next_observation = np.copy(self._state)
        reward =  self.reward_state(self._state)
        # print 'hence the reward is = ', reward
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self):
        print('current state:', self._state)