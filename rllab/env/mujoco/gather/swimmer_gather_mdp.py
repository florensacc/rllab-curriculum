from rllab.env.mujoco import GatherMDP
from rllab.env.mujoco import SwimmerMDP


class SwimmerGatherMDP(GatherMDP):

    MODEL_CLASS = SwimmerMDP
    ORI_IND = 2
