from sandbox.carlos_snn.envs.mujoco.maze.fast_maze_env import FastMazeEnv  # %^&*&^%
from sandbox.carlos_snn.envs.mujoco.snake_env import SnakeEnv
from rllab.envs.normalized_env import normalize
from rllab.core.serializable import Serializable


class SnakeMazeEnv(FastMazeEnv, Serializable):

    # MODEL_CLASS = normalize(SwimmerEnv)
    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 3
    MAZE_MAKE_CONTACTS = True

    # this is not needed, but on without the stub method I can't run a SnakeMazeEnv anymore!
    # def __init__(self, **kwargs):
    #     Serializable.quick_init(self, locals())
    #     super(SnakeMazeEnv, self).__init__(**kwargs)

