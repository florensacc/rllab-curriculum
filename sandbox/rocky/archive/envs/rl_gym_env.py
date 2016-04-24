from __future__ import print_function
from __future__ import absolute_import

import rl_gym  # import en
import rl_gym.envs
import rl_gym.spaces
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete


# from rllab.spaces.discrete import Discrete


def convert_rl_gym_space(space):
    if isinstance(space, rl_gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, rl_gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class RLGymEnv(Env, Serializable):
    def __init__(self, env_name):
        Serializable.quick_init(self, locals())
        env, env_id = rl_gym.envs.make(env_name)
        self.env = env
        self.env_id = env_id

        self._observation_space = convert_rl_gym_space(env.observation_space)
        self._action_space = convert_rl_gym_space(env.action_space)
        self._horizon = rl_gym.envs.spec(env_name).max_steps

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)
