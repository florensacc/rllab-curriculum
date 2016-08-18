import numpy as np
import os.path as osp

from rllab.envs.proxy_env import ProxyEnv
from rllab.misc.overrides import overrides
from rllab.misc import logger
import os

from sandbox.haoran.model_trpo.code.utils import central_finite_difference_jacobian

class AnalyticEnv(ProxyEnv):
    """
    An environment wrapper that allows analytic computation of the dynamics f and rewards r, including their gradients.

    For now we assume the environment to be deterministic.
    """

    def __init__(self,wrapped_env,fd_step):
        """
        fd_step: step size used by the central finite difference
        """
        self._wrapped_env = wrapped_env
        self.fd_step = fd_step

    @property
    def _full_state(self):
        return self._wrapped_env._full_state

    @overrides
    def reset(self,state=None):
        return self._wrapped_env.reset(state)

    def f(self,state,action):
        if hasattr(self._wrapped_env,"f"):
            return self._wrapped_env.f(state,action)
        else:
            self._wrapped_env.reset(state)
            observation,reward,done,info = self._wrapped_env.step(action)
            next_state = self._wrapped_env._full_state
            return next_state

    def f_s(self,state,action):
        if hasattr(self._wrapped_env,"f_s"):
            return self._wrapped_env.f_s(state,action)
        else:
            f_s = central_finite_difference_jacobian(
                func=self.f,
                inputs=[state,action],
                output_dim=len(state),
                step=self.fd_step,
                wrt=0,
            )
            return f_s

    def f_a(self,state,action):
        if hasattr(self._wrapped_env,"f_a"):
            return self._wrapped_env.f_a(state,action)
        else:
            f_a = central_finite_difference_jacobian(
                func=self.f,
                inputs=[state,action],
                output_dim=len(state),
                step=self.fd_step,
                wrt=1,
            )
            return f_a

    def r(self,state,action):
        if hasattr(self._wrapped_env,"r"):
            return self._wrapped_env.r(state,action)
        else:
            self._wrapped_env.reset(state)
            observation,reward,done,info = self._wrapped_env.step(action)
            return reward

    def r_s(self,state,action):
        if hasattr(self._wrapped_env,"r_s"):
            return self._wrapped_env.r_s(state,action)
        else:
            r_s = central_finite_difference_jacobian(
                func=self.r,
                inputs=[state,action],
                output_dim=1,
                step=self.fd_step,
                wrt=0,
            )
            return r_s

    def r_a(self,state,action):
        if hasattr(self._wrapped_env,"r_a"):
            return self._wrapped_env.r_a(state,action)
        else:
            r_a = central_finite_difference_jacobian(
                func=self.r,
                inputs=[state,action],
                output_dim=1,
                step=self.fd_step,
                wrt=1,
            )
            return r_a
