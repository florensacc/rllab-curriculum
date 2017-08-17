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
    FILE = "disk_generate_states.xml"

    def __init__(self,
                 init_solved=False,
                 center_lim=0.4,
                 *args, **kwargs):
        # self.target_position = np.array((0, 0.3))  # default center
        # self.amount_moved = 0.1
        MujocoEnv.__init__(self, *args, **kwargs)
        Serializable.quick_init(self, locals())

        self.init_solved = init_solved
        self.center_lim = center_lim
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
    def set_kill_outside(self):
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
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = False
        reward = 0

        return Step(
            ob, reward, done,
        )


