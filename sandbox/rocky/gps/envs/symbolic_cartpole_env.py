from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.box2d.cartpole_env import CartpoleEnv
import theano.tensor as TT


class SymbolicCartpoleEnv(CartpoleEnv):

    def __init__(self, *args, **kwargs):
        super(SymbolicCartpoleEnv, self).__init__(*args, **kwargs)
        self.reset_range = 0.

    def reward_sym(self, obs_var, action_var):
        xpos = obs_var[0]
        xvel = obs_var[1]
        apos = obs_var[2]
        avel = obs_var[3]
        # done = TT.cast(TT.or_(TT.abs_(xpos) > self.max_cart_pos, TT.abs_(apos) > self.max_pole_angle), 'float32')
        # abs(self.cart.position[0]) > self.max_cart_pos or \
        #     abs(self.pole.angle) > self.max_pole_angle

        # notdone = 1 - done
        ucost = 1e-7 * (action_var ** 2).sum()
        xcost = TT.square(apos)#1 - TT.cos(apos)
        return - xcost - ucost#notdone * 10 - notdone * xcost - notdone * ucost
