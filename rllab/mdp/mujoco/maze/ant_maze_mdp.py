from rllab.mdp.mujoco.ant_mdp import AntMDP
from rllab.mdp.mujoco.maze.maze_mdp import MazeMDP


class AntMazeMDP(MazeMDP):

    MODEL_CLASS = AntMDP
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0
