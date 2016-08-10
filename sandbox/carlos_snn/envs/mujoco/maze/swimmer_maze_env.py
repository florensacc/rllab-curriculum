from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.core.serializable import Serializable


class SwimmerMazeEnv(MazeEnv, Serializable):

    # MODEL_CLASS = normalize(SwimmerEnv)
    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True

    # # this is not needed, but on without the stub method I can't run a SwimmerMazeEnv anymore!
    # def __init__(self, **kwargs):
    #     Serializable.quick_init(self, locals())
    #     super(SwimmerMazeEnv, self).__init__(**kwargs)

