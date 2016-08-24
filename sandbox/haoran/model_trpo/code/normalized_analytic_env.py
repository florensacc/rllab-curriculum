import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.envs.normalized_env import NormalizedEnv
from rllab.misc.overrides import overrides
from rllab.envs.base import Step


class NormalizedAnalyticEnv(NormalizedEnv, Serializable):
    def __init__(
            self,
            env,
            **kwargs
    ):
        NormalizedAnalyticEnv.__init__(self, env,**kwargs)
        Serializable.quick_init(self, locals())

        # not implemented yet
        assert (not self._normalize_obs)
        assert (not self._normalize_reward)

    def reset(self,**kwargs):
        ret = self._wrapped_env.reset(**kwargs)
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def f_a(self,**kwargs):

    def r_a(self,**kwargs):
        
