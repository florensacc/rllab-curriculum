from rllab.env.mujoco import GatherMDP
from rllab.env.mujoco import PointMDP


class PointGatherMDP(GatherMDP):

    MODEL_CLASS = PointMDP
    ORI_IND = 2
