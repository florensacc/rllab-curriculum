from rllab.mdp.mujoco_1_22.swimmer_mdp import SwimmerMDP
from rllab.mdp.mujoco_1_22.maze.maze_mdp import MazeMDP


class SwimmerMazeMDP(MazeMDP):

    MODEL_CLASS = SwimmerMDP
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True
