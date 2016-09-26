"""
Call the system's default to automatically hash the entire item, which we assume to be a numpy array.
"""
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
import numpy as np

class SystemHash(Hash):
    def __init__(self,item_dim, dim_key=128, bucket_sizes=None):
        # each column is a vector of uniformly random orientation
        self.projection_matrix = np.random.normal(size=(item_dim, dim_key))
        self.item_dim = item_dim

        self.dim_key = dim_key
        if bucket_sizes is None:
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        # precompute modulos of powers of 2
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T

        # the tables count the number of observed keys for each bucket
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)),dtype=int)

    def compute_keys(self, items):
        """
        Compute the keys for many items (row-wise stacked as a matrix)
        """
        # compute the signs of the dot products with the random vectors
        binaries = np.sign(np.asarray(items).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_keys(self, keys):
        """
        Increment hash table counts for many items (row-wise stacked as a matrix)
        """
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_keys(self, keys):
        """
        For each item, return the min of all counts from all buckets.
        """
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        counts = np.asarray(all_counts).min(axis=0)
        return counts

    def reset(self):
        self.tables = np.zeros(
            (len(self.bucket_sizes), np.max(self.bucket_sizes))
        )
