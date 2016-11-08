from sandbox.carlos_snn.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.carlos_snn.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.core.serializable import Serializable


class SwimmerGatherEnv(GatherEnv, Serializable):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
