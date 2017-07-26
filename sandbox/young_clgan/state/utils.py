import multiprocessing
from rllab.sampler.stateful_pool import singleton_pool
import scipy.spatial
import random
from rllab import spaces
from rllab.misc import logger
import sys
import os.path as osp

import matplotlib as mpl

mpl.use('Agg')
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import numpy as np
import scipy.misc
from sandbox.young_clgan.state.evaluator import parallel_map, disable_cuda_initializer


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

    def append(self, states, n_process=None):
        if self.states_transform:
            return self.append_states_transform(states)
        if len(states) > 0:
            states = np.array(states)
            logger.log("we are trying to append states: {}".format(states.shape))
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states = self._process_states(states)
            logger.log("after processing, we are left with : {}".format(states.shape))
            if n_process is None:
                n_process = singleton_pool.n_parallel
            elif n_process in [-1, 0]:
                n_process = 1
            if n_process > 1:
                states_per_process = states.shape[0] // singleton_pool.n_parallel
                list_of_states = [states[i * states_per_process: (i+1) * states_per_process, :] for i in range(n_process-1)]
                list_of_states.append(states[states_per_process * (n_process - 1):, :])
                states = parallel_map(self._select_states, list_of_states)
                states = np.concatenate(states)
            else:
                states = self._select_states(states)
            self.state_list.extend(states.tolist())
            return states

    def _select_states(self, states):
        # print('selecting states from ', states.shape)
        selected_states = states
        if self.distance_threshold is not None and self.distance_threshold > 0:
            if len(self.state_list) > 0:
                dists = scipy.spatial.distance.cdist(self.state_list, selected_states)
                indices = np.amin(dists, axis=0) > self.distance_threshold
                selected_states = selected_states[indices, :]
        # print('the selected states are: {}'.format(selected_states.shape))
        return selected_states

    def _process_states(self, states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        states = np.array(states)
        results = [states[0]]
        for i, state in enumerate(states[1:]):
            # print("analyzing state : ", i)
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


