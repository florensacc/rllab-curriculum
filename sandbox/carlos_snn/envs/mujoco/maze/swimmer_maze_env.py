from sandbox.carlos_snn.envs.mujoco.maze.fast_maze_env import FastMazeEnv
from sandbox.carlos_snn.envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerMazeEnv(FastMazeEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 3
    MAZE_MAKE_CONTACTS = True

