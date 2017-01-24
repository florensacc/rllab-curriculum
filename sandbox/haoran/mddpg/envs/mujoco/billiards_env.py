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

    @overrides
    def get_current_obs(self):
        return self._full_state

    @overrides
    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        reward = 0
        done = False
        return Step(next_obs, reward, done)

    @overrides
    def get_viewer(self, config=None):
        super().get_viewer(config=self.window_config)
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 20
        self.viewer.cam.elevation = -90 # look down
        return self.viewer
