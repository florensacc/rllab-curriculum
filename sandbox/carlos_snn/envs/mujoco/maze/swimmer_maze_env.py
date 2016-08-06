from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize


class SwimmerMazeEnv(MazeEnv):

    # MODEL_CLASS = normalize(SwimmerEnv)
    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True

