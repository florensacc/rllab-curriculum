import numpy as np
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable


class ReacherEnv(MujocoEnv, Serializable):

    FILE = 'reacher.xml'

    def __init__(self, should_render=False, *args, **kwargs):
        super(ReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.should_render = should_render

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.model.forward()
        ob = self._get_obs()
        done = False
        return ob, reward, done

    def reset_mujoco(self, init_state=None):
        qpos_init = self.init_qpos + \
                                np.random.uniform(size=self.init_qpos.shape, low=-0.1, high=0.1)
        qvel_init = self.init_qvel + \
                                   np.random.uniform(size=self.init_qvel.shape, low=-0.005, high=0.005)
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2: break
        qpos_init[-2:] = self.goal
        qvel_init[-2:] = 0
        self.model.data.qpos = qpos_init
        self.model.data.qvel = qvel_init

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
