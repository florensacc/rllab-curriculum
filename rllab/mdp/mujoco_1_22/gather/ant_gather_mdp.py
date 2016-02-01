from rllab.mdp.mujoco_1_22.gather.gather_mdp import GatherMDP
from rllab.mdp.mujoco_1_22.ant_mdp import AntMDP


class AntGatherMDP(GatherMDP):

    MODEL_CLASS = AntMDP
    ORI_IND = 6
