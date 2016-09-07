from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from sandbox.rocky.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from sandbox.davis.hashing.bonus_evaluators.discretizing_hashing_bonus_evaluator import DiscretizingHashingBonusEvaluator


class OneHotHashingBonusEvaluator(DiscretizingHashingBonusEvaluator):
    def __init__(self, env_spec, num_buckets, obs_lower_bounds, obs_upper_bounds, *args, **kwargs):
        """num_buckets can be a number or a vector. If it is a vector, discretizes each observation
        dimension with different num_buckets.
        """
        HashingBonusEvaluator.__init__(self, env_spec, **kwargs)
        self.obs_lower_bounds = obs_lower_bounds - 1e-5  # So it can handle boundary values
        self.obs_upper_bounds = obs_upper_bounds + 1e-5
        obs_dim, dim_key = self.projection_matrix.shape
        if type(num_buckets) == int:
            self.num_buckets = np.repeat(num_buckets, obs_dim)
        assert len(self.num_buckets) == obs_dim
        self.projection_matrix = np.random.normal(size=(np.sum(self.num_buckets), dim_key))
        self.granularity = (obs_upper_bounds - obs_lower_bounds) / num_buckets.astype(float)

    def discretize(self, observations):
        num_obs, obs_dim = observations.shape
        int_discretized = np.floor((observations - self.obs_lower_bounds) / self.granularity)
        one_hot_discretized = np.zeros((num_obs, np.sum(self.num_buckets)))

        indices = np.roll(np.cumsum(self.num_buckets), 1)
        indices[0] = 0
        indices += int_discretized
        one_hot_discretized[np.repeat(np.arange(num_obs), obs_dim),
                            indices.flatten().astype(int)] = 1

        return one_hot_discretized
