from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab import spaces

from rllab.misc.overrides import overrides

BIG = 1e6

class MujocoEnv_ObsInit(MujocoEnv):

    def __init__(self, *args, **kwargs):
        super(MujocoEnv_ObsInit, self).__init__(*args, **kwargs)
        # quick fix
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        self._observation_space = spaces.Box(ub * -1, ub)

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    def get_ori(self):
        raise NotImplementedError
