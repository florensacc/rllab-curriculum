from sandbox.carlos_snn.envs.mujoco.follow.follow_env import FollowEnv
from sandbox.carlos_snn.envs.mujoco.snake_env import SnakeEnv


class SnakeFollowEnv(FollowEnv):
    MODEL_CLASS = SnakeEnv
    ORI_IND = 2

