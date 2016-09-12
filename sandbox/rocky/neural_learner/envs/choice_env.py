from rllab.envs.base import Env
from rllab.core.serializable import Serializable
import random


class ChoiceEnv(Env, Serializable):
    """
    Upon reset_trial, choose a random environment among a fixed set of choices. It is assumed that all environments
    should have the same observation space and action space
    """

    def __init__(self, envs):
        Serializable.quick_init(self, locals())
        self.envs = envs
        self.cur_env = None
        self.reset_trial()

    def reset_trial(self):
        self.cur_env = random.choice(self.envs)
        return self.cur_env.reset()

    def reset(self):
        return self.cur_env.reset()

    @property
    def observation_space(self):
        return self.cur_env.observation_space

    @property
    def action_space(self):
        return self.cur_env.action_space

    def step(self, action):
        return self.cur_env.step(action)
