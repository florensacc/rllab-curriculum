from rllab.env.mujoco.gather.gather_env import GatherEnv
from rllab.env.mujoco.swimmer_env import SwimmerEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
