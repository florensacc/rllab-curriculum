from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
import theano.tensor as TT
import numpy as np


class SymbolicCartpoleSwingupEnv(CartpoleSwingupEnv):

    def __init__(self, *args, **kwargs):
        super(SymbolicCartpoleSwingupEnv, self).__init__(*args, **kwargs)

        rng_state = np.random.get_state()
        np.random.seed(1)
        bounds = np.array([
            [-1, -2, np.pi-1, -3],
            [1, 2, np.pi+1, 3],
        ])
        low, high = bounds
        xpos, xvel, apos, avel = np.random.uniform(low, high)
        np.random.set_state(rng_state)
        self.reset_state = (xpos, xvel, apos, avel)

    def reward_sym(self, obs_var, action_var):
        xpos = obs_var[0]
        xvel = obs_var[1]
        apos = obs_var[2]
        avel = obs_var[3]
        ucost = 1e-7 * (action_var ** 2).sum()
        xcost = -TT.cos(apos)
        return - xcost - ucost

    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        xpos, xvel, apos, avel = self.reset_state
        self.cart.position = (xpos, self.cart.position[1])
        self.cart.linearVelocity = (xvel, self.cart.linearVelocity[1])
        self.pole.angle = apos
        self.pole.angularVelocity = avel
        return self.get_current_obs()

