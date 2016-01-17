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

    def step(self, state, action):
        prev_com = self.get_body_com("world")
        next_state = self.forward_dynamics(state, action, restore=False)
        after_com = self.get_body_com("world")

        next_obs = self.get_current_obs()
        # ctrl_cost = 1e-6 * np.sum(np.square(action))
        reward = (after_com[0] - prev_com[0]) / self.model.option.timestep
        # cost = ctrl_cost + run_cost
        # reward = -cost
        done = False
        return next_state, next_obs, reward, done
