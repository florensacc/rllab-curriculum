from rllab.env.mujoco import AntMDP
from rllab.env.mujoco import GatherMDP


class AntGatherMDP(GatherMDP):

    MODEL_CLASS = AntMDP
    ORI_IND = 6
