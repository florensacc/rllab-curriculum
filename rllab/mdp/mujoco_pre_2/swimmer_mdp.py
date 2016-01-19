from rllab.mdp.mujoco_pre_2.mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param))


class SwimmerMDP(MujocoMDP, Serializable):

    def __init__(self):
        path = self.model_path('swimmer.xml')
        frame_skip = 1
        ctrl_scaling = 1
        super(SwimmerMDP, self).__init__(path, frame_skip, ctrl_scaling)
        Serializable.__init__(self)

    def get_current_obs(self):
        qpos = self.model.data.qpos.flatten()
        qvel = self.model.data.qvel.flatten()
        return np.concatenate([qpos, qvel])

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
        prev_com = self.get_body_com("front")
        next_state = self.forward_dynamics(state, action, restore=False)
        after_com = self.get_body_com("front")

        next_obs = self.get_current_obs()
        ctrl_cost = 1e-5 * np.sum(np.square(action))
        run_cost = -1 * (after_com[0] - prev_com[0])# / self.model.option.timestep
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return next_state, next_obs, reward, done
