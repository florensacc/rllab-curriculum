from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
import numpy as np
import math
import random

from sandbox.young_clgan.lib.envs.base import GoalEnv
from sandbox.young_clgan.lib.envs.rewards import linear_threshold_reward


class PointEnv(GoalEnv, MujocoEnv, Serializable):

    FILE = 'point2.xml'

    def __init__(self, goal_generator, reward_dist_threshold=0.3,
            control_mode='linear', *args, **kwargs):

        self.control_mode = control_mode
        self.update_goal_generator(goal_generator)
        self.reward_dist_threshold = reward_dist_threshold
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        pos = self.model.data.qpos.flat[:2]
        vel = self.model.data.qvel.flat[:2]
        return np.concatenate([
            self.current_goal,
            pos,
            vel
        ])

    @overrides
    def reset(self):
        self.update_goal()
        qpos = np.zeros((self.model.nq, 1))
        qpos[2:, :] = np.array(self.current_goal).reshape((2, 1))
        qvel = np.zeros((self.model.nv, 1))
        self.set_state(qpos, qvel)
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def step(self, action):
        if self.control_mode == 'linear':
            self.forward_dynamics(action)
        elif self.control_mode == 'angular':
            vel = self.model.data.qvel.flat[:2]

            # Get the unit vector for velocity
            if np.linalg.norm(vel) < 1e-10:
                vel = np.array([1., 0.])
            else:
                vel = vel / np.linalg.norm(vel)
            acc = np.zeros_like(vel)
            acc += action[0] * vel
            acc += action[1] * np.array([-vel[1], vel[0]])
            self.forward_dynamics(acc)
        else:
            raise NotImplementedError("Control mode not supported!")

        reward_dist = self._compute_dist_reward()
        reward_ctrl = - np.square(action).sum()
        # reward = reward_dist + reward_ctrl
        reward = reward_dist

        dist = np.linalg.norm(
            self.get_body_com("torso") - self.get_body_com("target")
        )

        ob = self.get_current_obs()
        done = False
        return Step(
            ob, reward, done,
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            distance=dist,
            goal_distance=np.linalg.norm(np.array(self.current_goal)),
        )

    def _compute_dist_reward(self):
        dist = np.linalg.norm(
            self.get_body_com("torso") - self.get_body_com("target")
        )
        return linear_threshold_reward(dist, threshold=self.reward_dist_threshold, coefficient=-1000)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()

    @overrides
    def log_diagnostics(self, paths):
        # Process by time steps
        distances = [
            np.mean(path['env_infos']['distance'])
            for path in paths
        ]
        goal_distances = [
            np.mean(path['env_infos']['goal_distance'])
            for path in paths
        ]
        reward_dist = [
            np.mean(path['env_infos']['reward_dist'])
            for path in paths
        ]
        reward_ctrl = [
            np.mean(path['env_infos']['reward_ctrl'])
            for path in paths
        ]
        # Process by trajectories
        logger.record_tabular('GoalDistance', np.mean(goal_distances))
        logger.record_tabular('MeanDistance', np.mean(distances))
        logger.record_tabular('MeanRewardDist', np.mean(reward_dist))
        logger.record_tabular('MeanRewardCtrl', np.mean(reward_ctrl))
