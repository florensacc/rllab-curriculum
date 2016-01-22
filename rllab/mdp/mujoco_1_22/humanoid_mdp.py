from .mujoco_mdp import MujocoMDP
import numpy as np
from rllab.core.serializable import Serializable


# Taken from Wojciech's code
class HumanoidMDP(MujocoMDP, Serializable):

    FILE = 'humanoid.xml'

    def __init__(self, *args, **kwargs):
        super(HumanoidMDP, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

    def get_current_obs(self):
        data = self.model.data
        return np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            data.cinert.flat,
            data.cvel.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
        ])

    def _get_com(self):
        data = self.model.data
        mass = self.model.body_mass
        xpos = data.xipos
        return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

    def step(self, state, action):
        self.set_state(state)
        before_center = self._get_com()
        # self.model.forward()
        # before_com = self.get_body_com("front")
        next_state = self.forward_dynamics(state, action, restore=False)
        next_obs = self.get_current_obs()
        after_center = self._get_com()

        alive_bonus = 1.0
        data = self.model.data
        # mass = self.model.body_mass
        # xpos = data.xipos
        # after_center = (np.sum(mass * xpos, 0) / np.sum(mass))[0]
        lin_vel_cost = 0.25 * (after_center - before_center) / self.model.opt.timestep
        quad_ctrl_cost = .5e-4 * np.sum(np.square(data.ctrl))
        quad_impact_cost = .5e-4 * np.sum(np.square(data.cfrc_ext))
        quad_impact_cost = quad_impact_cost if quad_impact_cost < 10.0 else 10.0
        reward = lin_vel_cost + quad_ctrl_cost + quad_impact_cost + alive_bonus
        done = data.qpos[2] < 1.2 or data.qpos[2] > 2.0

        return next_state, next_obs, reward, done
