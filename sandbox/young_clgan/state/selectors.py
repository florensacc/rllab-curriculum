import random
from rllab import spaces
import sys
import os.path as osp

import matplotlib as mpl
mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
import tempfile
import math

from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from sandbox.young_clgan.envs.rewards import linear_threshold_reward


class StateSelector(object):
    """ Base class for state selector. """

    def __init__(self):
        self._state = None
        self.update()

    def update(self):
        return self.state

    @property
    def state(self):
        return self._state


class UniformListStateSelector(StateSelector, Serializable):
    """ Generating states uniformly from a state list. """

    def __init__(self, state_list):
        Serializable.quick_init(self, locals())
        self.state_list = state_list
        self.state_size = np.size(self.state_list[0])  # assumes all states have same dim as first in list
        random.seed()
        super(UniformListStateSelector, self).__init__()

    def update(self):
        self._state = random.choice(self.state_list)
        return self.state


class UniformStateSelector(StateSelector, Serializable):
    """ Generating states uniformly from a state list. """

    def __init__(self, state_size, bounds=2, center=()):
        Serializable.quick_init(self, locals())
        self.state_size = state_size
        self.bounds = bounds
        if np.array(self.bounds).size == 1:
            self.bounds = [-1 * bounds * np.ones(state_size), bounds * np.ones(state_size)]
        self.center = center if len(center) else np.zeros(self.state_size)
        super(UniformStateSelector, self).__init__()

    def update(self):  # This should be centered around the initial position!!
        sample = []
        for low, high in zip(*self.bounds):
            sample.append(np.random.uniform(low, high))
        self._state = self.center + np.array(sample)
        return self.state


class FixedStateSelector(StateSelector, Serializable):
    """ Generating a fixed state. """

    def __init__(self, state):
        Serializable.quick_init(self, locals())
        super(FixedStateSelector, self).__init__()
        self._state = state


def update_env_state_selector(env, state_selector):
    """ Update the state selector for normalized environment. """
    obj = env
    while not hasattr(obj, 'update_state_selector') and hasattr(obj, 'wrapped_env'):
        obj = obj.wrapped_env
    if hasattr(obj, 'update_state_selector'):
        return obj.update_state_selector(state_selector)
    else:
        raise NotImplementedError('Unsupported environment')

