from sandbox.davis.envs.point_env import PointEnv
from rllab.envs.base import Step
from rllab.core.serializable import Serializable
import numpy as np


class NoisyPointEnv(PointEnv, Serializable):
    def __init__(self, env_noise=0, reward_fn='norm', action_scale=1, *args, **kwargs):
        Serializable.quick_init(self, locals())
        PointEnv.__init__(self, *args, **kwargs)
        self.env_noise = env_noise
        self.reward_fn = reward_fn
        self.action_scale = action_scale

    def step(self, action):
        noise = np.random.normal(size=self._state.shape)
        self._state = self._state + action + self.env_noise * noise
        if self.reward_fn == 'norm':
            reward = -np.linalg.norm(self._state)
        elif self.reward_fn == 'noise':
            reward = -np.linalg.norm(noise)**2
        elif self.reward_fn == 'noise-action':
            reward = -np.linalg.norm(noise)**2 - self.action_scale * np.linalg.norm(action)**2
        done = self.end_early and np.all(self._state < 0.01)
        return Step(observation=np.copy(self._state), reward=reward, done=done, noise=noise)
