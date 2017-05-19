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
    """ Sliding block with wall """
    FILE = 'block_insertion_1.xml'


class BlockInsertionEnv2(BlockInsertionEnvBase):
    """ Sliding and rotating block with one wall """
    FILE = 'block_insertion_2.xml'
    
    
class BlockInsertionEnv3(BlockInsertionEnvBase):
    """ Sliding and rotating block with two walls """
    FILE = 'block_insertion_3.xml'



BLOCK_INSERTION_ENVS = [
    BlockInsertionEnv1,
    BlockInsertionEnv2,
    BlockInsertionEnv3,
]