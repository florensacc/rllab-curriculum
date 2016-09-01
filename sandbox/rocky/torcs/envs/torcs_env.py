import sys
from rllab.envs.base import Env, Step
from rllab.envs.gym_env import convert_gym_space
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
import numpy as np
import os
import atexit
sys.path.append('/opt/gym_torcs')

import gym_torcs

def cleanup():
    os.system('pkill torcs')
    os.system('pkill pulseaudio')

atexit.register(cleanup)

BIG = 50000

class TorcsEnv(Env, Serializable):

    def __init__(self):
        Serializable.quick_init(self, locals())
        self.env = env = gym_torcs.TorcsEnv(vision=False, throttle=False, xvfb=True)
        self._observation_space = Box(low=-BIG, high=BIG, shape=(68,))
        self._action_space = convert_gym_space(env.action_space)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        return self.convert_obs(self.env.reset(relaunch=True))

    def convert_obs(self, obs):
        return np.concatenate([obs.focus, [obs.speedX, obs.speedY, obs.speedZ], obs.opponents, [obs.rpm/1000], obs.track, obs.wheelSpinVel])

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return Step(observation=self.convert_obs(obs), reward=reward, done=done, **info)
