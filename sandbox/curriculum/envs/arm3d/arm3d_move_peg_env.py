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


class Arm3dMovePegEnv(MujocoEnv, Serializable):
    FILE = "arm3d_peg.xml"

    def __init__(self,
                 init_solved=False,
                 center_lim=0.4,
                 *args, **kwargs):
        self.target_position = np.array((0, 0.3))  # default center
        self.amount_moved = 0.1
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

        # self.init_qvel = np.zeros_like(self.init_qvel)
        # self.init_qacc = np.zeros_like(self.init_qacc)
        self.init_solved = init_solved
        self.center_lim = center_lim
        self.kill_outside = False
        # print("yo!")

    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,  # [:self.model.nq // 2],
            self.model.data.qvel.flat,  # [:self.model.nq // 2],
            # self.model.data.site_xpos[0],  # disc position
            self.target_position,
        ])

    @contextmanager
    def set_kill_outside(self):
        self.kill_outside = True
        try:
            yield
        finally:
            self.kill_outside = False

    def reset(self, init_state=None, target_position = None, *args, **kwargs):
        ret = super(Arm3dMovePegEnv, self).reset(init_state, *args, **kwargs)
        if not target_position:
            self.target_position = np.array((0 + random.uniform(-self.amount_moved, self.amount_moved),
                                             0.3 + random.uniform(-self.amount_moved, self.amount_moved)))
        else:
            self.target_position = target_position
        # self.model.data.site_xpos[1] = np.array([0, 0.35, 0])
        return ret

    def step(self, action):

        self.forward_dynamics(action)


        ob = self.get_current_obs()
        done = False
        curr = self.get_peg_position()
        reward = -np.linalg.norm(curr - self.target_position)
        # print("Target: {} Current: {}".format(self.target_position, curr))
        # set done to be safe
        if reward > -0.02:
            done = True

        return Step(
            ob, reward, done,
        )

    def get_peg_position(self):
        # self.get_body_com("peg")
        return self.model.data.xpos[-1][:2]

    def get_disc_position(self):
        return self.model.data.site_xpos[0]

    def get_goal_position(self):
        # return self.model.data.site_xpos[1]
        return self.model.data.xpos[-1] + np.array([0, 0, 0.05])  # this allows position to be changed todo: check this

    def get_vec_to_goal(self):
        disc_pos = self.get_disc_position()
        goal_pos = self.get_goal_position()
        return disc_pos - goal_pos  # note: great place for breakpoint!

    def get_distance_to_goal(self):
        vec_to_goal = self.get_vec_to_goal()
        return np.linalg.norm(vec_to_goal)

    def set_state(self, qpos, qvel):
        # assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()

    def log_diagnostics(self, paths):
        final_reward = [-path["rewards"][-1] for path in paths]
        logger.record_tabular('Final Distance', np.mean(final_reward))

