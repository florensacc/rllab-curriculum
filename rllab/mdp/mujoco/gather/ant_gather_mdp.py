from rllab.mdp.mujoco.gather.gather_mdp import GatherMDP
from rllab.mdp.mujoco.ant_mdp import AntMDP


class AntGatherMDP(GatherMDP):

    MODEL_CLASS = AntMDP
    ORI_IND = 6
