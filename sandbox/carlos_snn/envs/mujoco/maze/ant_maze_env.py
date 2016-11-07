from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv
from sandbox.carlos_snn.envs.mujoco.ant_env import AntEnv

from rllab.envs.normalized_env import normalize
from rllab.core.serializable import Serializable

class AntMazeEnv(MazeEnv, Serializable):

    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 3
    MAZE_SIZE_SCALING = 3.0
    # MAZE_MAKE_CONTACTS = True

