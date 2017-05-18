import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides

from sandbox.young_clgan.envs.goal_env import GoalEnv



class Reacher2DEnv(MujocoEnv, Serializable):

    FILE = 'reacher2d.xml'

    def __init__(self, *args, **kwargs):
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            self.model.data.geom_xpos[8, :2],
            self.model.data.qpos.flat[:2],
            np.cos(theta),
            np.sin(theta),
            self.model.data.qvel.flat[:2],
        ])

    @overrides
    def reset(self, **kwargs):
        if 'goal' in kwargs:
            goal = np.array(kwargs['goal']).flatten()
        else:
            goal = np.array([0, 0])
        qpos = np.random.uniform(low=-0.005, high=0.005, size=(self.model.nq, 1)) + self.init_qpos
        qpos[2:, 0] = goal
        qvel = self.init_qvel + np.random.uniform(low=-0.005, high=0.005, size=(self.model.nv, 1))
        qvel[2:, 0] = 0
        # import pdb; pdb.set_trace()
        self.set_state(qpos, qvel)
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        
        return self.get_current_obs()

    def step(self, action):
        self.forward_dynamics(action)
        reward = 0

        ob = self.get_current_obs()
        done = False
        return Step(
            ob, reward, done,
        )

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()
        
    def is_feasible(self, goal):
        return np.all(np.logical_and(-0.2 <= goal, goal <= 0.2))
