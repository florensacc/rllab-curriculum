


from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.spaces.box import Box
import theano.tensor as TT
import numpy as np

BIG = 1e6


class SymbolicSwimmerEnv(SwimmerEnv):
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_comvel("torso").flat,
            self.get_body_com("torso").flat,
        ]).reshape(-1)

    def reset_mujoco(self):
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

    def reward_sym(self, obs_var, action_var):
        forward_reward = obs_var[-6]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * TT.sum(
            TT.square(action_var / scaling))
        return forward_reward - ctrl_cost
