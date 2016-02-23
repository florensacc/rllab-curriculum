from rllab.mdp.mujoco.swimmer_mdp import SwimmerMDP
from rllab.mdp.mujoco.maze.maze_mdp import MazeMDP


class SwimmerMazeMDP(MazeMDP):

    MODEL_CLASS = SwimmerMDP
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True
