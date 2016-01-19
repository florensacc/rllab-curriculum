from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable


class Swimmer7MDP(MujocoMDP, Serializable):

    def __init__(self):
        path = self.model_path('swimmer7.xml')
        super(Swimmer7MDP, self).__init__(path, frame_skip=50, ctrl_scaling=1)
        Serializable.__init__(self)

    def get_current_obs(self):
        qpos = self.model.data.qpos.flatten()
        qvel = self.model.data.qvel.flatten()
        return np.concatenate([qpos, qvel])

    def step(self, state, action):
        #before_com = self.get_body_com("body")
        next_state = self.forward_dynamics(state, action, restore=False)

        #self.model.data.qvel[0]
        #after_com = self.get_body_com("body")

        #v = (after_com[0] - before_com[0]) / self.model.opt.timestep / self.frame_skip

        next_obs = self.get_current_obs()
        ctrl_cost = 0  # 1e-5 * np.sum(np.square(action))
        run_cost = -1 * self.model.data.qvel[0]
        cost = ctrl_cost + run_cost
        reward = -cost
        done = False
        return next_state, next_obs, reward, done
