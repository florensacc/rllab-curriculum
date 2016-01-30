from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class CartpoleMDP(MujocoMDP, Serializable):

    FILE = 'cartpole.xml'

    def __init__(
            self,
            *args, **kwargs):
        self.frame_skip = 1
        super(CartpoleMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(
            self, *args, **kwargs)

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1], # cart x pos
            np.sin(self.model.data.qpos[2:]), # link angles
            np.cos(self.model.data.qpos[2:]),
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)]
        ).reshape(-1)

    @overrides
    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_obs(next_state)
        x, _, y = self.model.data.site_xpos[0]
        r = -(0.01*x**2 + (y-2)**2)
        return next_state, next_obs, r, False

    @overrides
    def reset(self):
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self.current_state = self.get_current_state()
        return self.get_current_state(), self.get_current_obs()
