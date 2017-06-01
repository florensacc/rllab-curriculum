import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides


class BlockInsertionEnvBase(MujocoEnv, Serializable):
    
    FILE = None


    def __init__(self, *args, **kwargs):
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())
        self.file = self.__class__.FILE


    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[:self.model.nq // 2],
            self.model.data.qvel.flat[:self.model.nq // 2],
        ])

    @overrides
    def reset(self, **kwargs):
        # if 'goal' in kwargs:
        #     goal = np.array(kwargs['goal']).flatten()
        # else:
        #     goal = np.array([0, 0])
        init_qpos = np.copy(self.init_qpos)
        init_qvel = np.copy(self.init_qvel)
        init_qvel[:] = 0
        
        if 'goal' in kwargs and kwargs['goal'] is not None:
            init_qpos[self.model.nq // 2:, 0] = kwargs['goal']
        
        self.set_state(init_qpos, init_qvel)
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
        return np.all(np.logical_and(self.goal_lb <= goal, goal <= self.goal_ub))
        
    @property
    def goal_lb(self):
        return self.model.jnt_range[:self.model.nq // 2, 0]
        
    @property
    def goal_ub(self):
        return self.model.jnt_range[:self.model.nq // 2, 1]
        
    @property
    def goal_dim(self):
        return self.model.njnt // 2