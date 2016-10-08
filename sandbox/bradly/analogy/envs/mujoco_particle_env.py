from rllab.envs.base import Env, Step

from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
import numpy as np
import math
from rllab.mujoco_py import glfw
from rllab.mujoco_py import MjViewer


class Target:
    def __init__(self):
        self.x = 0
        self.y = 0


class Particle:
    def __init__(self):
        self.x = 0
        self.y = 0


class SimpleParticleEnv(MujocoEnv, Serializable):

    """
    Use Left, Right, Up, Down, A (steer left), D (steer right)
    """

    FILE = 'multi_modal_point.xml'

    def __init__(self, target_idxs=[0, 1], should_render=False,
                 init_height=50, init_width=50,
                 total_targets=4,
                 *args, **kwargs):

        self.total_targets_active = len(target_idxs)
        self.total_targets = total_targets
        self.target_idxs = target_idxs
        self.target_idx_current = 0
        super(SimpleParticleEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.should_render = should_render
        self.init_height = init_height
        self.init_width = init_width
        self.target = Target()
        self.particle = Particle()


    def get_reward(self):
        d_x = self.particle.x - self.target.x
        d_y = self.particle.y - self.target.y
        d_x_euclid = d_x*d_x
        d_y_euclid = d_y*d_y
        reward = math.sqrt(d_x_euclid + d_y_euclid)
        return -reward

    def step(self, action):
        if self.should_render is True:
            self.render()
        qpos = np.copy(self.model.data.qpos)
        qpos[2, 0] += action[1]
        ori = qpos[2, 0]
        # compute increment in each direction
        dx = math.cos(ori) * action[0]
        dy = math.sin(ori) * action[0]
        # ensure that the robot is within reasonable range
        qpos[0, 0] = np.clip(qpos[0, 0] + dx, -7, 7)
        qpos[1, 0] = np.clip(qpos[1, 0] + dy, -7, 7)
        self.model.data.qpos = qpos
        self.model.forward()
        self.maybe_switch_target(qpos)
        next_obs = self.get_current_obs()
        self.particle.x = qpos[0, 0]
        self.particle.y = qpos[1, 0]
        rew = self.get_reward()
        return Step(next_obs, rew, False)

    def maybe_switch_target(self, qpos):
        rew = -self.get_reward()
        if rew < 0.1:
            print('switching')
            self.target_idx_current += 1
            self.target_idx_current = min(self.target_idx_current, len(self.target_idxs) - 1)
            targ_idx_true = self.target_idxs[self.target_idx_current]
            targ = self.get_body_com("target_" + str(targ_idx_true))
            #print(targ)
            self.target.x = targ[0]
            self.target.y = targ[1]

    def get_xy(self):
        qpos = self.model.data.qpos
        return qpos[0, 0], qpos[1, 0]

    def set_xy(self, xy):
        qpos = np.copy(self.model.data.qpos)
        qpos[0, 0] = xy[0]
        qpos[1, 0] = xy[1]
        self.model.data.qpos = qpos
        self.model.forward()

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer(init_height=self.init_height, init_width=self.init_width)
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def reset_mujoco(self, init_state=None):
        qpos_init = self.init_qpos + \
                                np.random.uniform(size=self.init_qpos.shape, low=-0.5, high=0.5)
        qvel_init = self.init_qvel + \
                                   np.random.uniform(size=self.init_qvel.shape, low=-0.005, high=0.005)
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        #while True:
        #    self.goal = np.random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 2: break
        #point_locations = np.random.uniform(low=-.2, high=.2, size=2)
        #qpos_init[-12:] = self.goal
        qvel_init[-3*self.total_targets:] = 0
        self.model.data.qpos = qpos_init
        self.model.data.qvel = qvel_init
        #self.target_idxs = np.random.choice(np.arange(self.total_targets), self.total_targets_active)
        #self.target_idx_current = 0

    #def get_current_obs(self):
    #    targ = self.target_idxs[self.target_idx_current]
    #    basis_vec = np.zeros(shape=(self.total_targets,))
    #    basis_vec[targ] = 1
    #    targ_pos = self.get_body_com("target_" + str(0))
    #    return np.concatenate([
    #        targ_pos,
    #        basis_vec,
    #        self.model.data.qpos.flat,
    #        self.model.data.qvel.flat,
            #self.get_body_com("fingertip") - self.get_body_com("target")
    #    ])

    @overrides
    def action_from_key(self, key):
        lb, ub = self.action_bounds
        if key == glfw.KEY_LEFT:
            return np.array([0, ub[0]*0.3])
        elif key == glfw.KEY_RIGHT:
            return np.array([0, lb[0]*0.3])
        elif key == glfw.KEY_UP:
            return np.array([ub[1], 0])
        elif key == glfw.KEY_DOWN:
            return np.array([lb[1], 0])
        else:
            return np.array([0, 0])


