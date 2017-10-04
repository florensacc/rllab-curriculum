from curriculum.envs.maze.maze_env import MazeEnv
from curriculum.envs.maze.maze_swim.swimmer_env import SwimmerEnv


class SwimmerMazeEnv(MazeEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 6.0

    # MANUAL_COLLISION = True  # this is in point_mass
    MAZE_MAKE_CONTACTS = True  # this is in rllab
