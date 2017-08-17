import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from contextlib import contextmanager


class Arm3dDiscEnv(MujocoEnv, Serializable):
    FILE = "arm3d_disc.xml"

    def __init__(self,
                 init_solved=True,
                 kill_radius=0.4,
                 *args, **kwargs):
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

        # self.init_qvel = np.zeros_like(self.init_qvel)
        # self.init_qacc = np.zeros_like(self.init_qacc)
        self.init_solved = init_solved
        self.kill_radius = kill_radius
        self.kill_outside = False
        # print("yo!")


    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat, #[:self.model.nq // 2],
            self.model.data.qvel.flat, #[:self.model.nq // 2],
            self.model.data.site_xpos[0], # disc position
        ])

    @contextmanager
    def set_kill_outside(self, kill_outside=True, radius=None):
        self.kill_outside = True
        old_kill_radius = self.kill_radius
        if radius is not None:
            self.kill_radius = radius
        try:
            yield
        finally:
            self.kill_outside = False
            self.kill_radius = old_kill_radius

    def reset(self, init_state=None, *args, **kwargs):
        # init_state = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0) # TODO: used for debugging only!
        ret = super(Arm3dDiscEnv, self).reset(init_state, *args, **kwargs)
        # self.current_goal = self.model.data.geom_xpos[-1][:2]
        # print(self.current_goal) # I think this is the location of the peg
        return ret

    def step(self, action):
        # print(action.shape)
        self.forward_dynamics(action)
        distance_to_goal = self.get_distance_to_goal()
        reward = -distance_to_goal
        # print(self.model.data.site_xpos[1])
        # print(self.model.data.qpos[-2:])

        # if distance_to_goal < 0.03:
        #     print("inside the PR2DiscEnv, the dist is: {}, goal_pos is: {}".format(distance_to_goal, self.get_goal_position()))
            # print("Qpos: " + str(self.model.data.qpos))

        ob = self.get_current_obs()
        # print(ob)
        done = False

        if self.kill_outside and distance_to_goal > self.kill_radius:
            # print("******** OUT of region ********")
            done = True

        return Step(
            ob, reward, done,
        )

    # @overrides
    # def reset(self, **kwargs):
    #
    #     solved_qpos = [1.09725987, 0.70068112, -2.26654845, -1.65700369, 2.58752605, -2.12087312, 2.11059846]
    #
    #     # if 'goal' in kwargs:
    #     #     goal = np.array(kwargs['goal']).flatten()
    #     # else:
    #     #     goal = np.array([0, 0])
    #
    #     if self.init_solved is not None and self.init_solved:
    #         init_qpos = np.array(solved_qpos)
    #     else:
    #         init_qpos = np.copy(self.init_qpos)
    #     init_qvel = np.copy(self.init_qvel)
    #     init_qvel[:] = 0
    #
    #     # if 'goal' in kwargs and kwargs['goal'] is not None:
    #     #     init_qpos[self.model.nq // 2:, 0] = kwargs['goal']
    #
    #     self.set_state(init_qpos, init_qvel)
    #     self.current_com = self.model.data.com_subtree[0]
    #     self.dcom = np.zeros_like(self.current_com)
    #
    #     return self.get_current_obs()

    def get_disc_position(self):
        return self.model.data.site_xpos[0]

    def get_goal_position(self):
        # return self.model.data.site_xpos[1]
        return self.model.data.xpos[-1] + np.array([0, 0, 0.05]) # this allows position to be changed todo: check this

    def get_vec_to_goal(self):
        disc_pos = self.get_disc_position()
        goal_pos = self.get_goal_position()
        return disc_pos - goal_pos # note: great place for breakpoint!

    def get_distance_to_goal(self):
        vec_to_goal = self.get_vec_to_goal()
        return np.linalg.norm(vec_to_goal)


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