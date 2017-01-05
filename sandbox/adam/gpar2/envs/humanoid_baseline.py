"""
This is like Humanoid-v1, the gym environment, except this will skip gym and be
an rllab mujoco environment.

Will be the first one implemented with pre- allcating mujoco_py.

Maybe I should double check that this runs, first...OK it does.
"""

from rllab.envs.mujoco.mujoco_env import MujocoEnv
# from sandbox.adam.gpar.envs.mujoco_env_mine import MujocoEnv

import numpy as np
from rllab.core.serializable import Serializable
# from rllab.misc.overrides import overrides
# from rllab.misc import logger


def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class HumanoidEnv(MujocoEnv, Serializable):

    FILE = 'humanoid.xml'

    def __init__(
            self,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            *args, **kwargs):
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        super(HumanoidEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.frame_skip = 5  # For some reason this was getting set differently.

    # Borrowed from mujoco_env_gym
    def do_simulation(self, ctrl, n_frames):
        self.model.data.ctrl = ctrl
        for _ in range(n_frames):
            self.model.step()

    def get_current_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def step(self, action):
        pos_before = mass_center(self.model)
        self.do_simulation(action, self.frame_skip)
        pos_after = mass_center(self.model)
        alive_bonus = 5.0
        data = self.model.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.model.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self.get_current_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self.get_current_obs()


    # @overrides
    # def log_diagnostics(self, paths):
    #     progs = [
    #         path["observations"][-1][-3] - path["observations"][0][-3]
    #         for path in paths
    #     ]
    #     logger.record_tabular('AverageForwardProgress', np.mean(progs))
    #     logger.record_tabular('MaxForwardProgress', np.max(progs))
    #     logger.record_tabular('MinForwardProgress', np.min(progs))
    #     logger.record_tabular('StdForwardProgress', np.std(progs))
