from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import itertools
from rllab.misc import logger


class HashingBonusEvaluator(object):
    """
    Encode each observation as the signs of its dot products with random vectors (with uniformly sampled orientations). Then further compress the binary code as \sum_{i=0}^{dim_key} b_i 2^i (mod bucket_size) for each bucket. The bucket sizes are primes, so that we obtain distinct keys. Bucket keys are redundant to reduce the error caused by compression (see the query part).
    When a new observation is given, increment the key counts for all buckets.
    When querying the count of an observation, compute the key counts for all buckets but return the minimum. This way the count is less prone to errors caused by key compression.
    """
    def __init__(self, env_spec, dim_key=128, bucket_sizes=None):
        """
        dim_key: number of random vectors for projection
        """
        # Hashing function: SimHash
        if bucket_sizes is None:
            # some large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in xrange(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        # the tables count the number of observed keys for each bucket
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        obs_dim = env_spec.observation_space.flat_dim
        # each column is a vector of uniformly random orientation
        self.projection_matrix = np.random.normal(size=(obs_dim, dim_key))

    def compute_keys(self, observations):
        # denormalize: convert from [-1,1] to [0,255]
        observations = np.cast['int']((observations + 1) * 0.5 * 255.0)

        # compute the signs w.r.t. the random normal vectors
        binaries = np.sign(np.asarray(observations).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, observations):
        """
        Increment hash table counts
        """
        keys = self.compute_keys(observations)
        for idx in xrange(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_hash(self, observations):
        """
        Return the min of all counts from all buckets.
        """
        keys = self.compute_keys(observations)
        all_counts = []
        for idx in xrange(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        counts = np.asarray(all_counts).min(axis=0)
        return counts

    def fit_before_process_samples(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        self.inc_hash(observations)
        counts = self.query_hash(observations)
        logger.record_tabular('MinCount', np.min(counts))
        logger.record_tabular('MaxCount', np.max(counts))
        logger.record_tabular('AverageCount', np.mean(counts))
        logger.record_tabular('MedianCount', np.median(counts))
        logger.record_tabular('StdCount', np.std(counts))

    def predict(self, path):
        """
        Compute a bonus score.
        """
        counts = self.query_hash(path["observations"])
        return 1. / np.maximum(1., np.sqrt(counts))

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        pass
