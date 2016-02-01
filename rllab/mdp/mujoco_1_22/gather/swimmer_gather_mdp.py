from rllab.mdp.mujoco_1_22.gather.gather_mdp import GatherMDP
from rllab.mdp.mujoco_1_22.swimmer_mdp import SwimmerMDP


class SwimmerGatherMDP(GatherMDP):

    MODEL_CLASS = SwimmerMDP
    ORI_IND = 2
