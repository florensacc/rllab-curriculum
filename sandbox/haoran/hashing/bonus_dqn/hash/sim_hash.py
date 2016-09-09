from sandbox.haoran.hashing.bonus_dqn.hash.base import Hash
import numpy as np

class SimHash(Hash):
    def __init__(self,item_dim, dim_key=128, bucket_sizes=None):
        """
        Encode each item (vector) as the signs of its dot products with random vectors (with uniformly sampled orientations) to get a binary code. Then further compress the binary code as \sum_{i=0}^{dim_key} b_i 2^i (mod bucket_size) for each bucket. The bucket sizes are primes, so that we obtain distinct keys. Bucket keys are redundant to reduce the error caused by compression (see the query part).
        A key is finally represented as indices in the bukckets.
        When a new item is given, increment the key counts for all buckets.
        When querying the count of an item, compute the key counts for all buckets but return the minimum. This way the count is less prone to errors caused by key compression.
        """
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
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))

    def compute_keys(self, items):
        """
        Compute the keys for many items (row-wise stacked as a matrix)
        """
        # compute the signs of the dot products with the random vectors
        binaries = np.sign(np.asarray(items).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc(self, items):
        """
        Increment hash table counts for many items (row-wise stacked as a matrix)
        """
        keys = self.compute_keys(items)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query(self, items):
        """
        For each item, return the min of all counts from all buckets.
        """
        keys = self.compute_keys(items)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        counts = np.asarray(all_counts).min(axis=0)
        return counts

    def reset(self):
        self.tables = np.zeros(
            (len(self.bucket_sizes), np.max(self.bucket_sizes))
        )
