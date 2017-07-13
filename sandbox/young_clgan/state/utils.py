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

    def __init__(self, distance_threshold=None, states_transform = None):
        self.distance_threshold = distance_threshold
        self.state_list = []
        self.states_transform = states_transform
        if self.states_transform:
            self.transformed_state_list = []

    @property
    def size(self):
        return len(self.state_list)

    def empty(self):
        self.state_list = []

    def sample(self, size, replace=False, replay_noise=0):
        states = sample_matrix_row(np.array(self.state_list), size, replace)
        if replay_noise > 0:
            states += replay_noise * np.random.randn(*states.shape)
        return states

    def _process_states(self, states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        states = np.array(states)
        results = [states[0]]
        for state in states[1:]:
            if np.amin(scipy.spatial.distance.cdist(results, state.reshape(1, -1))) > self.distance_threshold:
                results.append(state)
        return np.array(results)

    def _process_states_transform(self, states, transformed_states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        results = [states[0]]
        transformed_results = [transformed_states[0]]
        for i in range(1, len(states)):
            # checks if valid in transformed space
            if np.amin(scipy.spatial.distance.cdist(transformed_results, transformed_states[i].reshape(1, -1))) > self.distance_threshold:
                results.append(states[i])
                transformed_results.append(transformed_states[i])
        return np.array(results), np.array(transformed_results)

    def append_states_transform(self, states):
        if len(states) > 0:
            states = np.array(states)
            transformed_states = self.states_transform(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states, transformed_states = self._process_states_transform(states, transformed_states)
                if len(self.state_list) > 0:
                    print("hi")
                    dists = scipy.spatial.distance.cdist(self.transformed_state_list, transformed_states)
                    indices = np.amin(dists, axis=0) > self.distance_threshold
                    states = states[indices, :]
                    transformed_states = transformed_states[indices, :]
            self.state_list.extend(states)
            self.transformed_state_list.extend(transformed_states)
            assert(len(self.state_list) == len(self.transformed_state_list))
            return states # modifed to return added states

    def append(self, states):
        if self.states_transform:
            return self.append_states_transform(states)
        if len(states) > 0:
            states = np.array(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states = self._process_states(states)
                if len(self.state_list) > 0:
                    dists = scipy.spatial.distance.cdist(self.state_list, states)
                    indices = np.amin(dists, axis=0) > self.distance_threshold
                    states = states[indices, :]
            self.state_list.extend(states)
            return states # modifed to return added states

    @property
    def states(self):
        return np.array(self.state_list)


def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0]:
        return M
    if replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size, replace=replace)
    return M[indices, :]


