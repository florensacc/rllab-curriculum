from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from sandbox.davis.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator


class DiscretizingHashingBonusEvaluator(HashingBonusEvaluator):
    def __init__(self, env_spec, granularity, *args, **kwargs):
        """Granularity can be a number or a vector. If it is a vector, discretizes each observation
        dimension with different granularity.
        """
        HashingBonusEvaluator.__init__(self, env_spec, **kwargs)
        self.granularity = granularity

    def compute_keys(self, observations):
        observations = self.discretize(observations)
        binaries = np.sign(np.asarray(observations).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def discretize(self, observations):
        return np.around(observations / self.granularity)
