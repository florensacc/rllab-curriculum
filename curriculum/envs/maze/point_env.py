from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import logger
import numpy as np
import math
import random

from curriculum.envs.base import StateGenerator
from curriculum.envs.goal_env import GoalEnv
from curriculum.envs.rewards import linear_threshold_reward


class PointEnv(GoalEnv, MujocoEnv, Serializable):
    FILE = 'point2.xml'

    def __init__(self,
                 goal_generator=None, reward_dist_threshold=0.1, indicator_reward=True, append_goal=False,
                 control_mode='linear',
                 *args, **kwargs):
        """
        :param goal_generator: Proceedure to sample and keep the goals
        :param reward_dist_threshold:
        :param control_mode:
        """
        Serializable.quick_init(self, locals())
        GoalEnv.__init__(self, goal_generator=goal_generator)
        self.control_mode = control_mode
        if goal_generator is None:
            self.update_goal_generator(StateGenerator())
        else:
            self.update_goal_generator(goal_generator)
        self.reward_dist_threshold = reward_dist_threshold
        self.indicator_reward = indicator_reward
        self.append_goal = append_goal
        MujocoEnv.__init__(self, *args, **kwargs)

    @overrides
    def get_current_obs(self):
        """Append obs with current_goal"""
        pos = self.model.data.qpos.flat[:2]
        vel = self.model.data.qvel.flat[:2]
        if self.append_goal:
            return np.concatenate([
                pos,
                vel,
                self.current_goal,
            ])
        else:
            return np.concatenate([pos, vel])

    @overrides
    def reset(self, init_state=None, goal=(1, 0), *args, **kwargs):  # reset called when __init__, so needs goal!
        # import pdb; pdb.set_trace()
        """This does both the reset of mujoco, the forward and reset goal"""
        self.update_goal(goal=goal)
        qpos = np.zeros((self.model.nq, 1))
        qvel = np.zeros((self.model.nv, 1))  # 0 velocity
        if init_state is not None:  # you can reset only the com position!
            qpos[:2] = np.array(init_state[:2]).reshape((2, 1))
            if np.array(init_state).size == 4:
                qvel[:2] = np.array(init_state[2:]).reshape((2, 1))
        qpos[2:, :] = np.array(self.current_goal).reshape((2, 1))  # the goal is part of the mujoco!!
        self.set_state(qpos, qvel)
        # this is usually the usual reset
        self.current_com = self.model.data.com_subtree[0]  # CF: this is very weird... gets 0, 2, 0.1 even when it's 0
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def step(self, action):
        # print('PointEnv, the action taken is: ', action)
        if self.control_mode == 'linear':  # action is directly the acceleration
            self.forward_dynamics(action)
        elif self.control_mode == 'angular':  # action[0] is accel in forward (vel) direction, action[1] in orthogonal.
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

        reward_dist = self._compute_dist_reward()  # 1000 * self.reward_dist_threshold at goal, decreases with 1000 coef
        reward_ctrl = - np.square(action).sum()
        # reward = reward_dist + reward_ctrl
        reward = reward_dist

        dist = np.linalg.norm(
            self.get_body_com("torso") - self.get_body_com("target")
        )

        ob = self.get_current_obs()
        # print('current obs:', ob)
        done = False
        if dist < self.reward_dist_threshold and self.indicator_reward:
            # print("**DONE***")
            done = True
        return Step(
            ob, reward, done,
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            distance=dist,
        )

    @overrides
    @property
    def goal_observation(self):  # transforms a state into a goal (projection, for example)
        return self.get_body_com("torso")[:2]

    def _compute_dist_reward(self):
        """Transforms dist to goal with linear_threshold_reward: gets -threshold * coef at dist=0, and decreases to 0"""
        dist = np.linalg.norm(
            self.get_body_com("torso") - self.get_body_com("target")
        )
        if self.indicator_reward and dist <= self.reward_dist_threshold:
            return 1000 * self.reward_dist_threshold
        else:
            return linear_threshold_reward(dist, threshold=self.reward_dist_threshold, coefficient=-1000)

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()

    def get_xy(self):
        qpos = self.model.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def set_xy(self, xy):
        qpos = np.copy(self.model.data.qpos)
        qpos[0, 0] = xy[0]
        qpos[1, 0] = xy[1]
        self.model.data.qpos = qpos
        self.model.forward()

    @overrides
    def log_diagnostics(self, paths):
        # Process by time steps
        distances = [
            np.mean(path['env_infos']['distance'])
            for path in paths
        ]
        goal_distances = [
            path['env_infos']['distance'][0] for path in paths
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
