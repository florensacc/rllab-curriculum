import scipy.spatial
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


class StateCollection(object):
    """ A collection of states, with minimum distance threshold for new states. """

    def __init__(self, distance_threshold=None):
        self.distance_threshold = distance_threshold
        self.state_list = []

    @property
    def size(self):
        return len(self.state_list)

    def sample(self, size, replace=False):
        return sample_matrix_row(np.array(self.state_list), size, replace)

    def _process_states(self, states):
        "keep only the states that are at more than dist_threshold from each other"
        states = np.array(states)

        results = [states[0]]
        for state in states[1:]:
            if np.amin(scipy.spatial.distance.cdist(results, state.reshape(1, -1))) > self.distance_threshold:
                results.append(state)
        return np.array(results)

    def append(self, states):
        if states:
            states = np.array(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states = self._process_states(states)
                if len(self.state_list) > 0:
                    dists = scipy.spatial.distance.cdist(self.state_list, states)
                    indices = np.amin(dists, axis=0) > self.distance_threshold
                    states = states[indices, :]
            self.state_list.extend(states)

    @property
    def states(self):
        return np.array(self.state_list)


def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0]:
        return M
    if replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size)
    return M[indices, :]
