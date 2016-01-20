from rllab.mdp.mujoco_pre_2.mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param))


class SwimmerMDP(MujocoMDP, Serializable):

    def __init__(self):
        path = self.model_path('icml-humanoid.xml')
        frame_skip = 1
        ctrl_scaling = 1
        super(SwimmerMDP, self).__init__(path, frame_skip, ctrl_scaling)
        Serializable.__init__(self)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten(),
            self.model.data.qvel.flatten(),
            self.model.data.cfrc_ext
            np.clip(self.model.data.qfrc_constraint.flatten(), -10, 10),
            self.get_body_com("torso"),
        ]).reshape(-1)
        # qpos = self.model.data.qpos.flatten()
        # qvel = self.model.data.qvel.flatten()
        # return np.concatenate([qpos, qvel])

    # def reset(self):
    #     qpos = self.init_qpos + np.random.uniform(-5, 5, self.init_qpos.shape)
    #     qvel = np.random.uniform(-0.1, 0.1, self.init_qvel.shape)
    #     self.model.data.qpos = qpos
    #     self.model.data.qvel = qvel
    #     self.model.data.ctrl = self.init_ctrl
    #     self.model.forward()
    #     self.current_state = self.get_current_state()
    #     return self.get_current_state(), self.get_current_obs()

    def step(self, state, action):
        self.set_state(state)
        com_before = self.get_body_com("torso")
        next_state = self.forward_dynamics(state, action, restore=False)
        com_after = self.get_body_com("torso")
        dcom = com_after - com_before
        comvel = dcom / self.model.opt.timestep / self.frame_skip
        forward_reward = 1 * comvel[0]
        velocity_deviation_cost = 1 * (comvel[1]**2 + comvel[2]**2)
        ctrl_cost = 0.5 * 1e-5 * np.sum(np.square(action / 100.0))
        impact_cost = min(0.5, 0.5 * 1e-5 * np.sum(np.square(self.model.data.qfrc_constraint)))
        survive_reward = self._survive_reward
        reward = forward_reward - ctrl_cost - impact_cost + survive_reward

        notdone = np.isfinite(next_state).all() and next_state[2] >= 0.2 and next_state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return next_state, ob, reward, done
