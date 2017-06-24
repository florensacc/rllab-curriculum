import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides


class DiscInsertionEnv(MujocoEnv, Serializable):
    
    FILE = "arm3d_disc.xml"

    def __init__(self,
                 init_solved=True,
                 *args, **kwargs):

        self.file = self.__class__.FILE
        self.init_solved = init_solved

        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat, #[:self.model.nq // 2],
            self.model.data.qvel.flat, #[:self.model.nq // 2],
        ])

    @overrides
    def reset(self, **kwargs):


        solved_qpos = [1.09725987, 0.70068112, -2.26654845, -1.65700369, 2.58752605, -2.12087312, 2.11059846]

        # if 'goal' in kwargs:
        #     goal = np.array(kwargs['goal']).flatten()
        # else:
        #     goal = np.array([0, 0])

        if self.init_solved is not None and self.init_solved:
            init_qpos = np.array(solved_qpos)
        else:
            init_qpos = np.copy(self.init_qpos)
        init_qvel = np.copy(self.init_qvel)
        init_qvel[:] = 0
        
        # if 'goal' in kwargs and kwargs['goal'] is not None:
        #     init_qpos[self.model.nq // 2:, 0] = kwargs['goal']
        
        self.set_state(init_qpos, init_qvel)
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        
        return self.get_current_obs()

    def get_disc_position(self):
        return self.model.data.site_xpos[0]

    def get_goal_position(self):
        return self.model.data.site_xpos[1]

    def get_vec_to_goal(self):
        disc_pos = self.get_disc_position()
        goal_pos = self.get_goal_position()
        return disc_pos - goal_pos

    def get_distance_to_goal(self):
        vec_to_goal = self.get_vec_to_goal()
        return np.linalg.norm(vec_to_goal)

    def step(self, action):
        self.forward_dynamics(action)
        distance_to_goal = self.get_distance_to_goal()
        reward = -distance_to_goal

        # if distance_to_goal < 0.03:
        #     print("Qpos: " + str(self.model.data.qpos))

        ob = self.get_current_obs()
        done = False
        
        return Step(
            ob, reward, done,
        )

    def set_state(self, qpos, qvel):
        #assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()
        
    # def is_feasible(self, goal):
    #     return np.all(np.logical_and(self.goal_lb <= goal, goal <= self.goal_ub))
    #
    # @property
    # def goal_lb(self):
    #     return self.model.jnt_range[:self.model.nq // 2, 0]
    #
    # @property
    # def goal_ub(self):
    #     return self.model.jnt_range[:self.model.nq // 2, 1]
    #
    # @property
    # def goal_dim(self):
    #     return self.model.njnt // 2