from rllab.mdp.mujoco_1_22.mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.overrides import overrides


class InvertedDoublePendulum(MujocoMDP, Serializable):

    FILE = 'inverted_double_pendulum.xml'

    @autoargs.arg("random_start", type=bool,
                  help="Randomized starting position by adjusting the angles"
                       "When this is false, the double pendulum started out"
                       "in balanced position")
    def __init__(
            self,
            *args, **kwargs):
        self.random_start = kwargs.get("random_start", True)
        super(InvertedDoublePendulum, self).__init__(*args, **kwargs)
        Serializable.__init__(
            self, *args, **kwargs)

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],  # cart x pos
            np.sin(self.model.data.qpos[1:]),  # link angles
            np.cos(self.model.data.qpos[1:]),
            np.clip(self.model.data.qvel, -10, 10),
            np.clip(self.model.data.qfrc_constraint, -10, 10)]
        ).reshape(-1)

    @overrides
    def step(self, state, action):
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_obs(next_state)
        x, _, y = self.model.data.site_xpos[0]
        dist_penalty = 0.01 * x ** 2 + (y - 2) ** 2
        v1, v2 = self.model.data.qvel[1:3]
        vel_penalty = 1e-3 * v1**2 + 5e-3 * v2**2
        alive_bonus = 10
        r = alive_bonus - dist_penalty - vel_penalty
        done = y <= 1
        return next_state, next_obs, r, done

    @overrides
    def reset(self):
        qpos = np.copy(self.init_qpos)
        if self.random_start:
            qpos[1] = (np.random.rand()-0.5)*40/180.*np.pi
        self.model.data.qpos = qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self.current_state = self.get_current_state()
        return self.get_current_state(), self.get_current_obs()

