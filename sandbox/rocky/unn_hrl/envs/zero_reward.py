from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step


class ZeroRewardEnv(ProxyEnv):
    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        return Step(obs, 0., done, **info)


zero_reward = ZeroRewardEnv
