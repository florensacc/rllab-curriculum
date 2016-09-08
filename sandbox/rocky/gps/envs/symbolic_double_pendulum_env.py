


from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from .symbolic_env import SymbolicEnv
from rllab.spaces.box import Box
import theano.tensor as TT
import numpy as np

BIG = 1e6


class SymbolicDoublePendulumEnv(DoublePendulumEnv, SymbolicEnv):
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        # stds = np.array([0.1, 0.1, 0.01, 0.01])
        pos1, pos2, v1, v2 = np.random.randn(4) * 0.005
        self.link1.angle = pos1
        self.link2.angle = pos2
        self.link1.angularVelocity = v1
        self.link2.angularVelocity = v2
        return self.get_current_obs()

    def reward_sym(self, obs_var, action_var):
        tgt_pos = np.asarray([0, self.link_len * 2])
        cur_pos = obs_var[-2:]
        dist = TT.sqrt(TT.sum(TT.square(cur_pos - tgt_pos)) + 1e-8)
        return -dist

    def get_current_obs(self):
        parent_obs = super(SymbolicDoublePendulumEnv, self).get_current_obs()
        tip_pos = self.get_tip_pos()
        return np.concatenate([parent_obs, tip_pos])

    @property
    def observation_space(self):
        parent_space = super(SymbolicDoublePendulumEnv, self).observation_space
        return Box(low=-BIG, high=BIG, shape=(parent_space.shape[0] + 2,))
