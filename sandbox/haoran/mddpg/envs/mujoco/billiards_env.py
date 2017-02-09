import numpy as np
import os

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.misc.overrides import overrides

class BilliardsEnv(MujocoEnv, Serializable):
    def __init__(
            self,
            **kwargs):
        FILE = os.path.join(
            os.path.dirname(__file__),
            "models",
            "billiards.xml"
        )
        super().__init__(file_path=FILE, **kwargs)
        self.frame_skip = 100
        Serializable.quick_init(self, locals())
        self.window_config = dict(
            title="billiards",
            xpos=0,
            ypos=0,
            width=500,
            height=500,
        )

    @overrides
    def get_current_obs(self):
        return self._full_state

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        n_billiards = 8
        qpos = self.model.data.qvel.ravel()
        pos_list = [
            qpos[i: i+2]
            for i in range(0, n_billiards * 7, 7)
        ]
        qvel = self.model.data.qvel
        vel_list = [
            qvel[i: i+2]
            for i in range(0, n_billiards * 6, 6)
        ]
        reward = np.sum([np.linalg.norm(vel) for vel in vel_list])
        print(reward)
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def get_viewer(self, config=None):
        super().get_viewer(config=self.window_config)
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 20
        self.viewer.cam.elevation = -90 # look down; must do this to avoid aliasing
        return self.viewer
