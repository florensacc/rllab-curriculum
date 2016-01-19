from mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np

class AntMDP(MujocoMDP, Serializable):

    def __init__(self, horizon=250, timestep=0.01):
        frame_skip = 1
        ctrl_scaling = 100.0
        self.timestep = timestep
        path = self.model_path('ant.xml')
        self.horizon = horizon
        super(AntMDP, self).__init__(path, frame_skip, ctrl_scaling)
        Serializable.__init__(self, horizon, timestep)
        init_qpos = np.zeros_like(self.model.data.qpos)
        # Taken from John's code
        init_qpos[0] = 0.0
        init_qpos[2] = 0.55
        init_qpos[8] = 1.0
        init_qpos[10] = -1.0
        init_qpos[12] = -1.0
        init_qpos[14] = 1.0
        self.init_qpos = init_qpos

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos[:1],
            self.model.data.qpos[2:],
            np.sign(self.model.data.qvel),
            np.sign(self.model.data.qfrc_constraint),
        ]).reshape(-1)

    # def get_current_com(self):
    #     xipos = self.model.data.xipos[1:]
    #     body_mass = self.model.body_mass[1:]
    #     return (xipos * body_mass).sum(axis=0) / body_mass.sum()

    def step(self, state, action):
        self.set_state(state)
        com_before = self.get_body_com("torso")
        next_state = self.forward_dynamics(state, action, restore=False)
        com_after = self.get_body_com("torso")
        forward_reward = com_after[0] - com_before[0]  # ) / self.timestep
        ctrl_cost = 1e-6 * np.sum(np.square(action))
        impact_cost = 1e-3 * np.sum(np.square(self.model.data.qfrc_impulse))
        reward = forward_reward - ctrl_cost - impact_cost + 0.05
        notdone = np.isfinite(next_state).all() and next_state[2] >= 0.2 and next_state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return next_state, ob, reward, done
