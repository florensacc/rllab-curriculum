from sandbox.young_clgan.lib.envs.maze_env import MazeEnv
from sandbox.young_clgan.lib.envs.point_env import PointEnv


class PointMazeEnv(MazeEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True
