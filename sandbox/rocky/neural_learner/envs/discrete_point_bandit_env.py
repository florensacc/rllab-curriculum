import numpy as np
from cached_property import cached_property

from rllab.envs.base import Env, Step
from rllab.spaces import Box
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.point_bandit_env import PointBanditEnv
from sandbox.rocky.tf.envs.vec_env import VecEnv


class DiscretePointBanditEnv(PointBanditEnv):

    @property
    def action_space(self):
        return Discrete(4)

