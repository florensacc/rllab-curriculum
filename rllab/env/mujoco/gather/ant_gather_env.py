from rllab.env.mujoco.gather.gather_env import GatherEnv
from rllab.env.mujoco.ant_env import AntEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
