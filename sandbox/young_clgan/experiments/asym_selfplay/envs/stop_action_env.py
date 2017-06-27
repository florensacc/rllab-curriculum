import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step


class StopActionEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)


    def reset(self, **kwargs):
        ret = self._wrapped_env.reset(**kwargs)
        return ret

    @property
    @overrides
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            wrapped_low = np.append(self._wrapped_env.action_space.low,[-1])
            wrapped_high =  np.append(self._wrapped_env.action_space.high, [1])
            return spaces.Box(wrapped_low, wrapped_high)
        else:
            raise NotImplementedError

    @overrides
    def step(self, action):
        wrapped_step = self._wrapped_env.step(action[:-1])
        next_obs, reward, done, info = wrapped_step
        if np.tanh(action[-1])>0.9:
            done = True
        else:
            done = False
        return Step(next_obs, reward, done, **info)

    def get_current_obs(self):
        return self._wrapped_env.get_current_obs()

    def __str__(self):
        return "Wrapped with stop action: %s" % self._wrapped_env

