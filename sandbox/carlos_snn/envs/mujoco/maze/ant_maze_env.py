# from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv
from sandbox.carlos_snn.envs.mujoco.maze.fast_maze_env import MazeEnv  # %^&*&^%
from sandbox.carlos_snn.envs.mujoco.ant_env import AntEnv
from rllab.misc.overrides import overrides

from rllab.envs.normalized_env import normalize
from rllab.core.serializable import Serializable

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import math


class AntMazeEnv(MazeEnv, Serializable):

    MODEL_CLASS = AntEnv
    ORI_IND = 3

    MAZE_HEIGHT = 3
    MAZE_SIZE_SCALING = 3.0
    # MAZE_MAKE_CONTACTS = True

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.wrapped_env.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori