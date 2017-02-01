from sandbox.carlos_snn.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.carlos_snn.envs.mujoco.ant_env import AntEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 3

