from rllab.env.mujoco import MazeMDP
from rllab.env.mujoco import SwimmerMDP


class SwimmerMazeMDP(MazeMDP):

    MODEL_CLASS = SwimmerMDP
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True
