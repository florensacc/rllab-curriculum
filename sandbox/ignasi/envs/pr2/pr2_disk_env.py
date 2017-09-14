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


class Pr2DiskEnv(MujocoEnv, Serializable):
    FILE = "pr2_disk.xml"

    def __init__(self,
                 init_solved=True,
                 kill_radius=0.4,
                 dist_weight=0,
                 ctrl_regularizer_weight=1,
                 action_torque_lambda=0,
                 disc_mass=0.15,
                 physics_variances=(0, 0, 0, 0),
                 start_peg=True,
                 start_dyn=True,
                 *args, **kwargs):
        """
        :param init_solved: 
        :param kill_radius: 
        :param dist_weight: 
        :param ctrl_regularizer_weight: 
        :param action_torque_lambda: 
        :param physics_variances: in this order -> damping, armature, friction, disk mass
        :param args: 
        :param kwargs: 
        """
        Serializable.quick_init(self, locals())
        self.init_solved = init_solved
        self.kill_radius = kill_radius
        self.kill_outside = False
        self.dist_weight = dist_weight
        self.ctrl_regularizer_weight = ctrl_regularizer_weight
        self.action_torque_lambda = action_torque_lambda
        self.disc_mass = disc_mass
        self.physics_variances = physics_variances
        self.start_peg = start_peg
        self.start_dyn = start_dyn

        MujocoEnv.__init__(self, *args, **kwargs)
        self.body_pos = self.model.body_pos.copy()
        # self.init_qvel = np.zeros_like(self.init_qvel)
        # self.init_qacc = np.zeros_like(self.init_qacc)
        self.frame_skip = 5

    @overrides
    def get_current_obs(self):
        joint_position = np.copy(self.model.data.qpos.flat)
        _, joint_position[4] = np.unwrap([0, joint_position[4]])
        _, joint_position[6] = np.unwrap([0, joint_position[6]])
        return np.concatenate([
            joint_position,  # [:self.model.nq // 2],
            self.model.data.qvel.flat,  # [:self.model.nq // 2],
            self.model.data.site_xpos[0],  # disc position
        ])

    def get_current_dyn(self):
        damping = (self.model.dof_damping[:, 0]).copy()
        armature = (self.model.dof_armature[:, 0]).copy()
        frictionloss = (self.model.dof_frictionloss[:, 0]).copy()
        id_tool = self.model.body_names.index('gear')
        xfrc = np.copy(self.model.data.xfrc_applied)
        mass = xfrc[id_tool, 2] / (-9.81)
        disc_mass = np.array([mass])
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

    @contextmanager
    def set_randomization(self, physics_variances=np.zeros(4,)):
        curr_physics_var = self.physics_variances.copy()
        self.physics_variances = physics_variances
        try:
            yield
        finally:
            self.physics_variances = curr_physics_var


    @property
    def start_observation(self):
        joint_position = self.get_current_obs()[:7]
        goal_xy = self.get_goal_position(relative=True)[:2]
        if self.start_dyn:
            dynamics = self.get_current_dyn()
            return np.concatenate([joint_position, goal_xy, dynamics])
        else:
             return np.concatenate([joint_position, goal_xy])

    def reset(self, init_state=None, *args, **kwargs):
        # if randomize:
        #   init_state = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0) # TODO: used for debugging only!
        dim = len(self.init_damping)
        damping = np.random.uniform(np.maximum(0., self.init_damping - self.physics_variances[0]),
                                    self.init_damping + self.physics_variances[0], dim)
        armature = np.random.uniform(np.maximum(0., self.init_armature - self.physics_variances[1]),
                                     self.init_armature + self.physics_variances[1], dim)
        frictionloss = np.random.uniform(np.maximum(0., self.init_frictionloss - self.physics_variances[2]),
                                         self.init_frictionloss + self.physics_variances[2], dim)

        self.model.dof_damping = damping[:, None]
        self.model.dof_frictionloss = frictionloss[:, None]
        self.model.dof_armature = armature[:, None]
        xfrc = np.zeros_like(self.model.data.xfrc_applied)
        id_tool = self.model.body_names.index('gear')
        xfrc[id_tool, 2] = - 9.81 * np.random.uniform(max(0., self.disc_mass - self.physics_variances[3]),
                                                      self.disc_mass + self.physics_variances[3])
        self.model.data.xfrc_applied = xfrc

        if init_state is not None:
            xfrc = np.zeros_like(self.model.data.xfrc_applied)
            id_tool = self.model.body_names.index('gear')
            if self.start_dyn and len(init_state) > 9:
                #I don't know why but the (x,y) position of the peg is set after the dyn
                id_start = 9 if self.start_peg else 7
                self.model.dof_damping = np.maximum(0, init_state[id_start:(id_start + 7)])
                self.model.dof_armature = np.maximum(0, init_state[(id_start + 7):(id_start + 14)])
                self.model.dof_frictionloss = np.maximum(0, init_state[(id_start + 14):(id_start + 21)])
                xfrc[id_tool, 2] = -9.81 * np.maximum(0, init_state[(id_start + 21)])
            else:
                self.model.dof_damping = damping[:, None]
                self.model.dof_armature = armature[:, None]
                self.model.dof_frictionloss = frictionloss[:, None]

            self.model.data.xfrc_applied = xfrc


            # hack if generated states don't have peg position
            if len(init_state) == 7:
                x = random.random() * 0.1
                y = random.random() * 0.1
                init_state.extend([x, y])

            # sets peg to desired position
            # print(init_state)
            pos = self.body_pos.copy()
            pos[-2, 0] += init_state[7]
            pos[-2, 1] += init_state[8]
            self.model.body_pos = pos
            init_state = init_state[:7]  # sliced so that super reset can reset joints correctly

        ret = super(Pr2DiskEnv, self).reset(init_state, *args, **kwargs)
        # print(self.get_goal_position())
        # geom_pos = self.model.body_pos.copy()
        # geom_pos[-2,:] += np.array([0,0,10])
        # self.model.body_pos = geom_pos
        # self.current_goal = self.model.data.geom_xpos[-1][:2]
        # print(self.current_goal) # I think this is the location of the peg
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
        dynamics = self.get_current_dyn()
        done = False
        # if distance_to_goal < 0.3:
        #     print("dist_to_goal: {}, rew: {}, next_obs: {}".format(distance_to_goal, reward, ob))

        if self.kill_outside and (distance_to_goal > self.kill_radius):
            print("******** OUT of region ********")
            done = True

        return Step(
            ob, reward, done, distance=distance_to_goal, goal_relative=goal_relative, ctrl_penalty=ctrl_penalty,
            dynamics=dynamics,
        )

    def get_disc_position(self):
        id_gear = self.model.body_names.index('gear')
        return self.model.data.xpos[id_gear]

    def get_goal_position(self, relative=False):
        if relative:
            return self.model.data.site_xpos[-1] - np.array([0.4146814, 0.47640087, 0.5305665])  # todo: not hardcode this?
        else:
            return self.model.data.site_xpos[-1]  # note, slightly different from previous, set to bottom of peg

    def get_vec_to_goal(self):
        disc_pos = self.get_disc_position()
        goal_pos = self.get_goal_position()
        # print("disc pos: {}, goal_pos: {}".format(disc_pos, goal_pos))
        return disc_pos - goal_pos

    def get_distance_to_goal(self):
        vec_to_goal = self.get_vec_to_goal()
        return np.linalg.norm(vec_to_goal)

    def set_state(self, qpos, qvel):
        # assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
        # print('SET STATE')
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()

    def transform_to_start_space(self, obs, env_infos):  # hard-coded that the first 7 coord are the joint pos.
        if self.start_dyn:
            return np.concatenate([obs[:7], env_infos['goal_relative'][:2], env_infos['dynamics']])  # using 'goal' takes the one from the goal_env
        else:
            return np.concatenate([obs[:7], env_infos['goal_relative'][:2]])  # using 'goal' takes the one from the goal_env
        # remove the last one, it's the z coordinate of the peg and it doesn't move.


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
