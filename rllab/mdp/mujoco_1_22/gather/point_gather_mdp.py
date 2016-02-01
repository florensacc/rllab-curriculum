from rllab.mdp.mujoco_1_22.gather.gather_mdp import GatherMDP
from rllab.mdp.mujoco_1_22.point_mdp import PointMDP


class PointGatherMDP(GatherMDP):

    MODEL_CLASS = PointMDP
    ORI_IND = 2
