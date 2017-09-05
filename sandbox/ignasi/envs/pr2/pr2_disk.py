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
      self.model.data.qpos.flat,  # [:self.model.nq // 2],
      self.model.data.qvel.flat,  # [:self.model.nq // 2],
      # self.model.data.site_xpos[0], # disc position
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

  @property
  def start_observation(self):
    return np.copy(self.model.data.qpos).flatten()

  def reset(self, init_state=None, *args, **kwargs):
    # if randomize:
    #   init_state = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0) # TODO: used for debugging only!
    dim = len(self.init_damping)
    damping = np.maximum(0, np.random.multivariate_normal(self.init_damping, 2 * np.eye(dim)))
    armature = np.maximum(0, np.random.multivariate_normal(self.init_armature, 2 * np.eye(dim)))
    frictionloss = np.maximum(0, np.random.multivariate_normal(self.init_frictionloss, 2 * np.eye(dim)))
    self.model.dof_damping = damping[:, None]
    self.model.dof_frictionloss = frictionloss[:, None]
    self.model.dof_armature = armature[:, None]
    # xfrc = np.zeros_like(self.model.data.xfrc_applied)
    # id_tool = self.model.body_names.index('gear')
    # xfrc[id_tool, 2] = - 9.81 * np.random.uniform(0.05, 0.5)
    # self.model.data.xfrc_applied = xfrc
    ret = super(Pr2DiskEnv, self).reset(init_state, *args, **kwargs)
    # self.current_goal = self.model.data.geom_xpos[-1][:2]
    # print(self.current_goal) # I think this is the location of the peg
    return ret

  def step(self, action):
    # print(action.shape)
    self.forward_dynamics(action)
    distance_to_goal = self.get_distance_to_goal()
    reward = -distance_to_goal
    ob = self.get_current_obs()
    done = False
    print("dist_to_goal: {}, rew: {}, next_obs: {}".format(distance_to_goal, reward, ob))

    if self.kill_outside and (distance_to_goal > self.kill_radius):
      print("******** OUT of region ********")
      done = True

    return Step(
      ob, reward, done, distance=distance_to_goal
    )

  def get_disc_position(self):
    id_gear = self.model.body_names.index('gear')
    return self.model.data.xpos[id_gear]

  def get_goal_position(self):
    return np.array([0.4146814, 0.47640087, 0.5305665])

  def get_vec_to_goal(self):
    disc_pos = self.get_disc_position()
    goal_pos = self.get_goal_position()
    print("disc pos: {}, goal_pos: {}".format(disc_pos, goal_pos))
    return disc_pos - goal_pos  # note: great place for breakpoint!

  def get_distance_to_goal(self):
    vec_to_goal = self.get_vec_to_goal()
    return np.linalg.norm(vec_to_goal)

  def set_state(self, qpos, qvel):
    # assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
    print('SET STATE')
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