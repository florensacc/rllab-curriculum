from rllab.env.mujoco import AntMDP
from rllab.env.mujoco import MazeMDP


class AntMazeMDP(MazeMDP):

    MODEL_CLASS = AntMDP
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0
