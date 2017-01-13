from sandbox.carlos_snn.envs.mujoco.gather.gather_env import GatherEnv
from sandbox.carlos_snn.envs.mujoco.snake_env import SnakeEnv
from rllab.core.serializable import Serializable


class SnakeGatherEnv(GatherEnv, Serializable):

    MODEL_CLASS = SnakeEnv
    ORI_IND = 2
