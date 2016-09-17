from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from sandbox.davis.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator
from sandbox.davis.hashing.bonus_evaluators.discretizing_hashing_bonus_evaluator import DiscretizingHashingBonusEvaluator


class OneHotHashingBonusEvaluator(DiscretizingHashingBonusEvaluator):
    def __init__(self,
                 env_spec,
                 num_buckets,
                 obs_lower_bounds=None,
                 obs_upper_bounds=None,
                 bounding_strategy='coerce',
                 **kwargs):
        """num_buckets can be a number or a vector. If it is a vector, discretizes each observation
        dimension with different num_buckets.

        bounding_strategy is what to do if something falls outside of the bounds:
            coerce: put it in the outermost bucket
            enforce: throw an AssertionError
            expand: holding num_buckets constant, change bounds to account for new data
        """
        HashingBonusEvaluator.__init__(self, env_spec, **kwargs)
        self.bounding_strategy = bounding_strategy
        obs_dim, dim_key = self.projection_matrix.shape
        if type(num_buckets) == int:
            self.num_buckets = np.repeat(num_buckets, obs_dim)
        assert len(self.num_buckets) == obs_dim
        self.projection_matrix = np.random.normal(size=(np.sum(self.num_buckets), dim_key))
        if obs_lower_bounds is not None and obs_upper_bounds is not None:
            self.initialize_buckets(obs_lower_bounds, obs_upper_bounds)
        else:
            self.initialized = False

    def initialize_buckets(self, obs_lower_bounds, obs_upper_bounds):
        self.obs_lower_bounds = obs_lower_bounds - 1e-5  # So it can handle boundary values
        self.obs_upper_bounds = obs_upper_bounds + 1e-5
        self.granularity = (obs_upper_bounds - obs_lower_bounds) / self.num_buckets.astype(float)
        self.initialized = True

    def discretize(self, observations):
        num_obs, obs_dim = observations.shape
        int_discretized = np.floor((observations - self.obs_lower_bounds) / self.granularity)
        if self.bounding_strategy == 'coerce':
            int_discretized = np.clip(int_discretized, 0, self.num_buckets - 1).astype(int)
        else:
            assert np.all(0 <= int_discretized) and np.all(int_discretized < self.num_buckets)
            # This will always be true in expand bounding_strategy
        one_hot_discretized = np.zeros((num_obs, np.sum(self.num_buckets)))

        # import pdb; pdb.set_trace()
        indices = np.roll(np.cumsum(self.num_buckets), 1)
        indices[0] = 0
        indices = indices + int_discretized
        one_hot_discretized[np.repeat(np.arange(num_obs), obs_dim),
                            indices.flatten().astype(int)] = 1

        return one_hot_discretized

    def fit_before_process_samples(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        if not self.initialized:
            self.initialize_buckets(observations.min(axis=0), observations.max(axis=0))
        elif self.bounding_strategy == 'expand':
            self.initialize_buckets(np.minimum(self.obs_lower_bounds, observations.min(axis=0)),
                                    np.maximum(self.obs_upper_bounds, observations.max(axis=0)))

        super(OneHotHashingBonusEvaluator, self).fit_before_process_samples(paths)
