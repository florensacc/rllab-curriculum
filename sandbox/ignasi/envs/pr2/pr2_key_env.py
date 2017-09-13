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


class Pr2KeyEnv(MujocoEnv, Serializable):
    FILE = "pr2_key.xml"

    def __init__(self,
                 init_solved=True,
                 kill_radius=0.4,
                 dist_weight = 0,
                 ctrl_regularizer_weight=1,
                 action_torque_lambda=1,
                 disc_mass=0.1,
                 physics_variances=(0, 0, 0, 0),
                 start_peg=True,
                 start_dyn=True,
                 *args, **kwargs):
        MujocoEnv.__init__(self, *args, **kwargs)
        self.frame_skip = 5
        Serializable.quick_init(self, locals())

        self.init_solved = init_solved
        self.kill_radius = kill_radius
        self.dist_weight = dist_weight
        self.ctrl_regularizer_weight = ctrl_regularizer_weight
        self.action_torque_lambda = action_torque_lambda
        self.kill_outside = False
        self.disc_mass = disc_mass
        self.physics_variances = physics_variances
        self.start_peg = start_peg
        self.start_dyn = start_dyn

        self.body_pos = self.model.body_pos.copy()
        self.frame_skip = 5 #todo: just set!

    @overrides
    def get_current_obs(self):
        joint_position = np.copy(self.model.data.qpos.flat)
        _, joint_position[4] = np.unwrap([0, joint_position[4]])
        _, joint_position[6] = np.unwrap([0, joint_position[6]])
        return np.concatenate([
            self.model.data.qpos.flat,  # [:self.model.nq // 2],
            self.model.data.qvel.flat,  # [:self.model.nq // 2],
            self.model.data.site_xpos[0],  # disc position, possibly remove?
        ])

    def get_current_dyn(self):
        damping = (self.model.dof_damping[:, 0]).copy()
        armature = (self.model.dof_armature[:, 0]).copy()
        frictionloss = (self.model.dof_frictionloss[:, 0]).copy()
        disc_mass = np.array([self.disc_mass])
        return np.concatenate([damping, armature, frictionloss, disc_mass])

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

    @property
    def start_observation(self):
        joint_position = self.get_current_obs()[:7]
        # return joint_position # change back!
        goal_xy = self.get_goal_position(relative=True)[:2]
        return np.concatenate([joint_position, goal_xy])

    def reset(self, init_state=None, *args, **kwargs):
        # best initial state
        # init_state = [0.34396303529542571, 0.36952090462532139, 1.2508105774646641, -1.8499649619190317,
        #               -4.4254893018593906, -1.9586739159844251, -1.3942096934113373]


        if init_state is not None:
            xfrc = np.zeros_like(self.model.data.xfrc_applied)
            id_tool = self.model.body_names.index('key')
            pos = self.body_pos.copy()
            pos[-2, 0] += init_state[-2]
            pos[-2, 1] += init_state[-1]
            self.model.body_pos = pos
            init_state = init_state[:7]
            if self.start_dyn and len(init_state) > 9:
                #I don't know why but the (x,y) position of the peg is set after the dyn
                id_start = 9 if self.start_peg else 7
                self.model.dof_damping = np.maximum(0, init_state[id_start:(id_start + 7)])
                self.model.dof_armature = np.maximum(0, init_state[(id_start + 7):(id_start + 14)])
                self.model.dof_frictionloss = np.maximum(0, init_state[(id_start + 14):(id_start + 21)])
                xfrc[id_tool, 2] = -9.81 * np.maximum(0, init_state[(id_start + 21)])
            else:
                dim = len(self.init_damping)
                damping = np.maximum(0, np.random.multivariate_normal(self.init_damping,
                                                                      self.physics_variances[0] * np.eye(dim)))
                armature = np.maximum(0, np.random.multivariate_normal(self.init_armature,
                                                                       self.physics_variances[1] * np.eye(dim)))
                frictionloss = np.maximum(0, np.random.multivariate_normal(self.init_frictionloss,
                                                                           self.physics_variances[2] * np.eye(dim)))
                self.model.dof_damping = damping[:, None]
                self.model.dof_armature = armature[:, None]
                self.model.dof_frictionloss = frictionloss[:, None]
                xfrc[id_tool, 2] = - 9.81 * np.maximum(0,
                                                       self.disc_mass + np.random.uniform(0, self.physics_variances[3]))
            self.model.data.xfrc_applied = xfrc
        ret = super(Pr2KeyEnv, self).reset(init_state, *args, **kwargs)


        return ret

    def step(self, action):
        # print(action.shape)
        self.forward_dynamics(action)
        distance_to_goal = self.get_distance_to_goal()
        goal_relative = self.get_goal_position(relative=True)
        # penalty for torcs:
        action_norm = np.linalg.norm(action)
        velocity_norm = np.linalg.norm(self.model.data.qvel)
        ctrl_penalty = - self.ctrl_regularizer_weight * (self.action_torque_lambda * action_norm + velocity_norm)
        reward = ctrl_penalty - self.dist_weight * distance_to_goal
        ob = self.get_current_obs()
        done = False
        # if distance_to_goal < 0.3:
        #     print("dist_to_goal: {}, rew: {}, next_obs: {}".format(distance_to_goal, reward, ob))

        if self.kill_outside and (distance_to_goal > self.kill_radius):
            print("******** OUT of region ********")
            done = True

        return Step(
            ob, reward, done, distance=distance_to_goal, goal_relative=goal_relative, ctrl_penalty=ctrl_penalty,
        )

    def get_disc_position(self):
        return self.model.data.site_xpos[0]

    def get_goal_position(self, relative = False):
        if relative:
            return self.model.data.site_xpos[-1] - np.array([0.417326, 0.0693085, 0.47])
        return self.model.data.site_xpos[-1]

    def get_vec_to_goal(self):
        disc_pos = self.get_disc_position()
        goal_pos = self.get_goal_position()
        # print("disc pos: {}, goal_pos: {}".format(disc_pos, goal_pos))
        return disc_pos - goal_pos  # note: great place for breakpoint!

    def get_distance_to_goal(self):
        vec_to_goal = self.get_vec_to_goal()
        return np.linalg.norm(vec_to_goal)

    def transform_to_start_space(self, obs, env_infos):  # hard-coded that the first 7 coord are the joint pos.
        return np.concatenate([obs[:7], env_infos['goal_relative'][:2]])  # using 'goal' takes the one from the goal_env
        # remove the last one, it's the z coordinate of the peg and it doesn't move.

    #
    def set_state(self, qpos, qvel):
        # assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        # print('SET STATE')
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()
