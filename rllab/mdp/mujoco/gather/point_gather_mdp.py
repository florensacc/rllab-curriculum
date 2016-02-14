from rllab.mdp.mujoco.gather.gather_mdp import GatherMDP
from rllab.mdp.mujoco.point_mdp import PointMDP


class PointGatherMDP(GatherMDP):

    MODEL_CLASS = PointMDP
    ORI_IND = 2
