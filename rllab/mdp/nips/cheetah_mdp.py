from rllab.mdp.nips.mujoco_mdp import MujocoMDP
from rllab.core.serializable import Serializable
import numpy as np


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param))


class CheetahMDP(MujocoMDP, Serializable):

    def __init__(self):
        path = self.model_path('cheetah.xml')
        frame_skip = 1
        ctrl_scaling = 1
        super(CheetahMDP, self).__init__(path, frame_skip, ctrl_scaling)
        Serializable.__init__(self)

    def get_current_obs(self):
        qpos = self.model.data.qpos.flatten()
        qvel = self.model.data.qvel.flatten()
        return np.concatenate([qpos[1:], qvel])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, state, action):
        prev_com = self.get_body_com("torso")
        next_state = self.forward_dynamics(state, action, restore=False)
        after_com = self.get_body_com("torso")

        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * np.sum(np.square(action))
        passive_cost = 1e-7 * np.sum(np.square(self.model.data.qfrc_passive))
        run_cost = -1 * (after_com[0] - prev_com[0]) / self.model.option.timestep
        upright_cost = 1e-5 * smooth_abs(self.get_body_xmat("torso")[2, 2], 0.1)
        cost = ctrl_cost + passive_cost + run_cost + upright_cost
        reward = -cost
        done = False
        return next_state, next_obs, reward, done
