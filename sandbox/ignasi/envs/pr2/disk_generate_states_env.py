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


class DiskGenerateStatesEnv(MujocoEnv, Serializable):
    FILE = "pr2_disk_w_joints.xml"

    def __init__(self,
                 init_solved=False,
                 evaluate = False, # allows for rollouts over generated states
                 kill_radius=0.4, #TODO: can tune!
                 kill_peg_radius = 0.1, # TODO: can tune!
                 *args, **kwargs):
        # self.amount_moved = 0.1
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

        self.original_position = np.copy(self.model.data.xpos[-1][:2]) # should be (0, 0.3)
        self.evaluate = evaluate
        self.init_solved = init_solved
        self.kill_radius = kill_radius
        self.kill_peg_radius = kill_peg_radius
        self.kill_outside = False

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # [:self.model.nq // 2],
            self.model.data.qvel.flat,  # [:self.model.nq // 2],
            # self.model.data.site_xpos[0],  # disc position
            # self.target_position,
        ])

    @contextmanager
    def set_kill_outside(self, *args, **kwargs):
        self.kill_outside = True
        try:
            yield
        finally:
            self.kill_outside = False

    @property
    def start_observation(self):
        return np.copy(self.model.data.qpos).flatten()

    def reset(self, init_state=None, *args, **kwargs):
        # init_state = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0)
        ret = super(DiskGenerateStatesEnv, self).reset(init_state, *args, **kwargs)
        return ret

    def step(self, action):

        # done is always False
        if self.evaluate:
            action = np.zeros_like(action)
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = False
        distance_to_goal = self.get_distance_to_goal()
        reward = 0
        # print(self.get_peg_displacement())
        # print(distance_to_goal, self.get_peg_displacement()) # useful for debugging
        if self.kill_outside and (
                distance_to_goal > self.kill_radius or self.get_peg_displacement() > self.kill_peg_radius):
            # print("******** OUT of region ********")
            done = True

        return Step(
            ob, reward, done,
        )

    #todo: should be good
    def get_peg_displacement(self):
        # geom of peg moves
        self.curr_position = np.array(self.model.data.xpos[-1][:2])
        return np.linalg.norm(self.curr_position - self.original_position)

    #todo: should be good
    def get_disc_position(self):
        # return self.model.data.geom_xpos[-1] - np.array([0, 0, 0.0125])
        # return self.model.data.site_xpos[0]
        id_gear = self.model.body_names.index('gear')
        return self.model.data.xpos[id_gear]

    #todo: should be good
    def get_goal_position(self):
        # return self.model.data.site_xpos[1] # old goal positions, should be (0, 0.3, -0.4)
        return self.model.data.geom_xpos[-1] - np.array([0, 0, 0.0125])
        # return self.model.data.xpos[-1] + np.array([0, 0, 0.05]) # this allows position to be changed todo: check this

    def get_vec_to_goal(self):
        disc_pos = self.get_disc_position()
        goal_pos = self.get_goal_position()
        return disc_pos - goal_pos # note: great place for breakpoint!

    def get_distance_to_goal(self):
        vec_to_goal = self.get_vec_to_goal()
        return np.linalg.norm(vec_to_goal)


