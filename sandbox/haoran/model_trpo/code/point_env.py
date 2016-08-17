from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np
import matplotlib.pyplot as plt


class PointEnv(Env):
    def __init__(self):
        self.init_state = np.array([1,1])

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def reset(self,state=None):
        if state is None:
            self._state = np.array([1,1])
        else:
            self._state = state
        observation = np.copy(self._state)
        return observation

    @property
    def _full_state(self):
        return self._state

    def step(self, action):
        s = self._state
        a = action
        next_s = self.f(s,a)
        reward = self.r(s,a)
        done = np.linalg.norm(next_s) < 0.01
        self._state = next_s
        return Step(observation=next_s, reward=reward, done=done)

    def render(self):
        print 'current state:', self._state

    def r(self,s,a):
        return -np.sum((s+a)** 2.)
    def r_s(self,s,a):
        return -(s+a)
    def r_a(self,s,a):
        return -(s+a)

    def f(self,s,a):
        return s+a
    def f_s(self,s,a):
        return np.eye(2)
    def f_a(self,s,a):
        return np.eye(2)

    def plot(self,path):
        states = path["observations"]
        xx = states[:,0]
        yy = states[:,1]
        plt.plot(xx,yy)
