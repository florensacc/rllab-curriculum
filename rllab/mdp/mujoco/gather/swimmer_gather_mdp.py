from rllab.mdp.mujoco.gather.gather_mdp import GatherMDP
from rllab.mdp.mujoco.swimmer_mdp import SwimmerMDP


class SwimmerGatherMDP(GatherMDP):

    MODEL_CLASS = SwimmerMDP
    ORI_IND = 2
