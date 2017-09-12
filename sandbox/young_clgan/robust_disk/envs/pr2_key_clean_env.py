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

  FILE = "pr2_key_find_init.xml"

  def __init__(self,
               init_solved=True,
               kill_radius=0.4,
               *args, **kwargs):
    MujocoEnv.__init__(self, *args, **kwargs)
    self.frame_skip = 5
    Serializable.quick_init(self, locals())

    self.init_solved = init_solved
    self.kill_radius = kill_radius
    self.kill_outside = False
    self.body_pos = self.model.body_pos.copy()

  @overrides
  def get_current_obs(self):
    return np.concatenate([
      self.model.data.qpos.flat,     # [:self.model.nq // 2],
      self.model.data.qvel.flat,     # [:self.model.nq // 2],
      self.model.data.site_xpos[0],  # disc position
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
    # init_state = [ 0.46808831,  0.68477385,  1.22024562, -2.32114235, -2.58833853, -2.07115401, -2.49469938] # very close
    #even warmer
    # init_state = [0.3326968224334913, 0.51069807363422348, 0.91457796350298493, -1.623275354231555, -3.9850499577912193, -2.0506470318735297, -1.6932902946856201]

    # very close to parallel
    # init_state = [0.34853855873484002, 0.37303888703108479, 1.2605103343415729, -1.8974411314510942, -4.4413078760856184, -1.9653540688328985, -1.4956710631295072]

    # parallel (best)
    init_state = [0.34396303529542571, 0.36952090462532139, 1.2508105774646641, -1.8499649619190317, -4.4254893018593906, -1.9586739159844251, -1.3942096934113373]

    # init_state = [ 1.38781535, -0.2317441, 2.65237236, -1.94273868, 4.78109335,-0.90467269, -1.56926878]
    # init_state = [1.2216135759588189, -0.52360156360043431, 2.3835233005680774, -2.0034129651264809, 4.4187603231907362, 2.1186197187178173e-05, -1.5864904744727759]
    # dim = len(self.init_damping)
    # damping = np.maximum(0, np.random.multivariate_normal(self.init_damping, 2 * np.eye(dim)))
    # armature = np.maximum(0, np.random.multivariate_normal(self.init_armature, 2 * np.eye(dim)))
    # frictionloss = np.maximum(0, np.random.multivariate_normal(self.init_frictionloss, 2 * np.eye(dim)))
    # self.model.dof_damping = damping[:, None]
    # self.model.dof_frictionloss = frictionloss[:, None]
    # self.model.dof_armature = armature[:, None]
    # xfrc = np.zeros_like(self.model.data.xfrc_applied)
    # id_tool = self.model.body_names.index('gear')
    # xfrc[id_tool, 2] = - 9.81 * np.random.uniform(0.05, 0.5)
    # self.model.data.xfrc_applied = xfrc

    # if init_state is not None:
    #   # hack if generated states don't have peg position
    #   if len(init_state) == 7:
    #     x = random.random() * 0.1
    #     y = random.random() * 0.1
    #     init_state.extend([x, y])
    #
    #   # sets peg to desired position
    #   print(init_state)
    #   pos = self.body_pos.copy()
    #   pos[-2, 0] += init_state[-2]
    #   pos[-2, 1] += init_state[-1]
    #   self.model.body_pos = pos
    #   init_state = init_state[:7] # sliced so that super reset can reset joints correctly
    ret = super(Pr2KeyEnv, self).reset(init_state, *args, **kwargs)
    # sets gravity
    # xfrc = np.zeros_like(self.model.data.xfrc_applied)
    # id_tool = self.model.body_names.index('keyhole')
    # xfrc[id_tool, 0] = 9.81 * 0.005 # moves away from robot
    # xfrc[id_tool, 1] =  -9.81 * 0.003 # moves parallel to robot
    # xfrc[id_tool, 2] = - 9.81 * 0.01 #gravity
    # xfrc[id_tool, 3] = 0.03 #rotates clockwise
    # xfrc[id_tool, 4] = 0.04
    # self.model.data.xfrc_applied = xfrc

    # print(self.get_goal_position())
    # geom_pos = self.model.body_pos.copy()
    # geom_pos[-2,:] += np.array([0,0,10])
    # self.model.body_pos = geom_pos
    # self.current_goal = self.model.data.geom_xpos[-1][:2]
    # print(self.current_goal) # I think this is the location of the peg
    return ret

  def step(self, action):

    # action = np.zeros_like(action)
    # print(action.shape)
    self.forward_dynamics(action)
    distance_to_goal = 0 # delete
    # distance_to_goal = self.get_distance_to_goal()
    reward = -distance_to_goal
    # reward = - np.linalg.norm(self.model.data.qpos) * 1e-2 - (self.model.data.site_xpos[0][2] - 0.47) ** 2
         # abs(self.model.data.site_xpos[0][0] - 0.55) ** 2
    ob = self.get_current_obs()
    done = False

    # if (self.model.data.site_xpos[0][2] - 0.47) < 0.005:
    #     print(list(self.model.data.qpos.flatten()))
    #     raise Exception

    # print("dist_to_goal: {}, rew: {}, next_obs: {}".format(distance_to_goal, reward, ob))
    #
    # if self.kill_outside and (distance_to_goal > self.kill_radius):
    #   print("******** OUT of region ********")
    #   done = True

    # Uncomment lines below to help get initial positoin
    # print(list(self.model.data.qpos.flatten())) # for getting initial position
    # print(reward)
    return Step(
      ob, reward, done, distance=distance_to_goal
    )

  # def get_disc_position(self):
  #   id_gear = self.model.body_names.index('gear')
  #   return self.model.data.xpos[id_gear]
  #
  # def get_goal_position(self):
  #   return self.model.data.site_xpos[-1] # note, slightly different from previous, set to bottom of peg
  #   # return np.array([0.4146814, 0.47640087, 0.5305665])
  #
  # def get_vec_to_goal(self):
  #   disc_pos = self.get_disc_position()
  #   goal_pos = self.get_goal_position()
  #   # print("disc pos: {}, goal_pos: {}".format(disc_pos, goal_pos))
  #   return disc_pos - goal_pos  # note: great place for breakpoint!
  #
  # def get_distance_to_goal(self):
  #   vec_to_goal = self.get_vec_to_goal()
  #   return np.linalg.norm(vec_to_goal)
  #
  # def set_state(self, qpos, qvel):
  #   # assert qpos.shape == (self.model.nq, 1) and qvel.shape == (self.model.nv, 1)
  #   # print('SET STATE')
  #   self.model.data.qpos = qpos
  #   self.model.data.qvel = qvel
  #   # self.model._compute_subtree() #pylint: disable=W0212
  #   self.model.forward()
