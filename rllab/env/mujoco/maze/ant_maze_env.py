from rllab.env.mujoco.maze.maze_env import MazeEnv
from rllab.env.mujoco.ant_env import AntEnv


class AntMazeEnv(MazeEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

