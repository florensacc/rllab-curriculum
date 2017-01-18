from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import os
import numpy as np

class ReacherEnv(MujocoEnv, Serializable):
    """
    A copy of OpenAI Gym's Reacher-v1, adapted for RLLab
    An additional termination condition is given
    """
    FILE = "reacher.xml"

    def __init__(
            self,
            deterministic=False,
            dist_threshold=0.01,
            *args,
            **kwargs
        ):
        """
        :param dist_threshold: if the dist to goal is under the threshold,
            terminate
        :param deterministic: fix the initial reacher position and velocity and
            the goal position
        """
        self.frame_skip = 2 # default in OpenAI
        self.deterministic = deterministic
        self.dist_threshold = dist_threshold
        super(ReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def render(self,close=False):
        viewer = self.get_viewer()
        viewer.cam.trackbodyid = 0
        viewer.loop_once()
        if close:
            self.stop_viewer()

    def step(self, action):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        dist = np.linalg.norm(vec)
        reward_dist = - dist
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + reward_ctrl
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = dist < self.dist_threshold
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    @overrides
    def reset_mujoco(self, init_state=None):
        if self.deterministic:
            qpos = np.copy(self.init_qpos)
        else:
            qpos = self.init_qpos + np.random.uniform(
                low=-0.1, high=0.1, size=(self.model.nq, 1))

        while True:
            if self.deterministic:
                self.goal = np.array([[0.], [0.19]])
            else:
                self.goal = np.random.uniform(low=-.2, high=.2, size=(2, 1))
            if np.linalg.norm(self.goal) < 2: break
        qpos[-2:] = self.goal
        if self.deterministic:
            qvel = np.copy(self.init_qvel)
        else:
            qvel = self.init_qvel + \
                np.random.uniform(low=-.005, high=.005, size=(self.model.nv, 1))
        qvel[-2:] = 0
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        return self.get_current_obs()

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
