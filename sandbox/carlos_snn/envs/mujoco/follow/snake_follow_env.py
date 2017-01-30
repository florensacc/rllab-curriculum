from rllab.core.serializable import Serializable
from sandbox.carlos_snn.envs.mujoco.follow.follow_env import FollowEnv
from sandbox.carlos_snn.envs.mujoco.snake_env import SnakeEnv
from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import math


class SnakeFollowEnv(FollowEnv, Serializable):
    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

