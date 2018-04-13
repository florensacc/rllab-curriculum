from curriculum.envs.maze.maze_env import MazeEnv
from curriculum.envs.maze.maze_ant.ant_env import AntEnv
# from curriculum.envs.maze.maze_ant.ant_target_env import AntEnv


class AntMazeEnv(MazeEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0 # should be 2.0?

    # MANUAL_COLLISION = True  # this is in point_mass
    # MAZE_MAKE_CONTACTS = True  # this is in rllab
