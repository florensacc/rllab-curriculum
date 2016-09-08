


from rllab.envs.box2d.cartpole_env import CartpoleEnv
from .symbolic_env import SymbolicEnv
import theano.tensor as TT


class SymbolicCartpoleEnv(CartpoleEnv, SymbolicEnv):
    def __init__(self, *args, **kwargs):
        CartpoleEnv.__init__(self, *args, **kwargs)
        self.reset_range = 0.

    def reward_sym(self, obs_var, action_var):
        xpos = obs_var[0]
        xvel = obs_var[1]
        apos = obs_var[2]
        avel = obs_var[3]
        ucost = 1e-7 * (action_var ** 2).sum()
        xcost = TT.square(apos)
        return - xcost - ucost
