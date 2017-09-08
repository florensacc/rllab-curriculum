from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs


class PR2KeyEnv(MujocoEnv, Serializable):
    FILE = 'pr2_arm3d_key.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(
            self,
            ctrl_cost_coeff=1e1,
            goal_dist=3e-2,
            *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        super(PR2KeyEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.goal_dist = goal_dist
        self.ee_indices = [14,23]
        self.frame_skip = 1
        self.init_qpos = np.array([0.1, 0.1, -1.54, -1.7, 1.54, -0.2, 0])

        theta = -np.pi / 2
        d = 0.15
        self.goal_position = np.array(
            [0.0, 0.3, -0.45 - d, 0.0, 0.3, -0.15 - d, 0.0 + 0.15 * np.sin(theta), 0.3 + 0.15 * np.cos(theta),
             -0.3 - d])
        self.cost_params = {
            'wp': np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            'l1': 0.1,
            'l2': 10.0,
            'alpha': 1e-5}

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            # self.model.data.site_xpos.flat, #todo: check what this is
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        # todo check which object has to be at goal position
        # todo also check the meaning of alpha
        # key_position = self.get_body_com('key_head1')
        ee_position = next_obs[self.ee_indices[0]:self.ee_indices[1]]
        dist = np.sum(np.square(self.goal_position - ee_position) * self.cost_params['wp'])
        dist_cost = np.sqrt(dist) * self.cost_params['l1'] + dist * self.cost_params['l2']
        # if dist < self.goal_dist:
        #     dist_cost = 0  # - dist
        # else:
        #     dist_cost = - 1.  # self.goal_dist
        reward = - dist_cost - ctrl_cost
        done = True if np.sqrt(dist) < self.goal_dist else False
        if done:
           print('MADE IT!')
           print(next_obs)
        return Step(next_obs, reward, done)
