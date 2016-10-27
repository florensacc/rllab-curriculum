import numpy as np
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable


class ReacherEnv(MujocoEnv, Serializable):

    FILE = 'multi_modal_reacher.xml'

    def __init__(self, should_render=False,
                 total_targets=2, target_idxs=[0, 1],
                 *args, **kwargs):
        self.total_targets = total_targets
        self.target_idxs = target_idxs
        self.current_target_idx = 0
        self.active_targets = len(self.target_idxs)
        super(ReacherEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        self.should_render = should_render

    def step(self, a):
        if self.should_render is True:
            self.render()
        vec = self.get_body_com("fingertip")-self.get_body_com("target_" + str(self.target_idxs[self.current_target_idx]))
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.model.forward()
        ob = self.get_current_obs()
        done = False
        return ob, reward, done, dict()

    def reset_mujoco(self, init_state=None):
        qpos_init = self.init_qpos + \
                                np.random.uniform(size=self.init_qpos.shape, low=-0.1, high=0.1)
        qvel_init = self.init_qvel + \
                                   np.random.uniform(size=self.init_qvel.shape, low=-0.005, high=0.005)
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl
        #while True:
        #    self.goal = np.random.uniform(low=-.2, high=.2, size=2)
        #    if np.linalg.norm(self.goal) < 2: break
        #qpos_init[-2:] = self.goal
        qvel_init[-3*self.total_targets:] = 0
        self.model.data.qpos = qpos_init
        self.model.data.qvel = qvel_init
        #self.current_target_idx = 0
        #self.target_idxs = np.random.choice(np.arange(self.total_targets), self.active_targets)

    def get_current_obs(self):
        theta = self.model.data.qpos.flat[:2]
        obs_vec = np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat, #[2:],
            self.model.data.qvel.flat, #[:2],
            self.get_body_com("fingertip") - self.get_body_com("target_" + str(self.target_idxs[self.current_target_idx]))
        ])
        #print(obs_vec.shape)
        return obs_vec
