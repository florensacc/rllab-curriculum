""" Just to add the method: weighted_sample_n """

from rllab.spaces.discrete import Discrete as BaseDiscrete
from rllab.misc import special
import numpy as np


class Discrete(BaseDiscrete):

    def __init__(self, n):
        self._n = n
        self._range_n_array = np.array(range(n))

    def weighted_sample_n(self, weights_matrix):
        return special.weighted_sample_n(weights_matrix, self._range_n_array)
