import random

import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides

from sandbox.young_clgan.envs.block_insertion.base import BlockInsertionEnvBase


class BlockInsertionEnv1(BlockInsertionEnvBase):

    FILE = 'block_insertion_1.xml'
    goal_lb = np.array([-0.36, -0.7, -0.3])
    goal_ub = np.array([0.36, 0, 0.3])


class BlockInsertionEnv2(BlockInsertionEnvBase):
    
    FILE = 'block_insertion_2.xml'
    goal_lb = np.array([-0.36, -0.7, -0.3, -3.14])
    goal_ub = np.array([0.36, 0, 0.3, 3.14])
