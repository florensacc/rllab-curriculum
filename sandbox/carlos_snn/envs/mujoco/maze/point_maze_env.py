from sandbox.carlos_snn.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.point_env import PointEnv


class PointMazeEnv(MazeEnv):
    MODEL_CLASS = PointEnv
    ORI_IND = 2

    MAZE_HEIGHT = 3  # this was 0.5
    MAZE_SIZE_SCALING = 4  # this was 4
    MAZE_MAKE_CONTACTS = True

