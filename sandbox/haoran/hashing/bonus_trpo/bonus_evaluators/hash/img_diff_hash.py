"""
Hash a sequence of frames
Use LSH for the first frame; then compute the pairwise binarized absolute difference for later frames, which is further hashed by some modulos.
Q: how to make the binarized difference stable to emulator noise / artifact?
"""

from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
import numpy as np

class ImgDiffHash(Hash):
    def __init__(
        self,
        frame_dim,
        n_frames,
        dim_key=128,
        first_bucket_sizes=None,
        other_bucket_sizes=None,
    ):
        # each column is a vector of uniformly random orientation
        self.projection_matrix = np.random.normal(size=(frame_dim, dim_key))
        self.frame_dim = frame_dim

        self.dim_key = dim_key
        if first_bucket_sizes is None:
            first_bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
        if other_bucket_sizes is None:
            other_bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]

        # precompute modulos of powers of 2
        first_mods_list = []
        for bucket_size in first_bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            first_mods_list.append(mods)
        self.first_bucket_sizes = np.asarray(first_bucket_sizes)
        self.first_mods_list = np.asarray(first_mods_list).T

        other_mods_list = []
        for bucket_size in other_bucket_sizes:
            mod = 1
            mods = []
            for _ in range(dim_key):
                mods.append(mod)
                mod = (mod * 2) % bucket_size
            other_mods_list.append(mods)
        self.other_bucket_sizes = np.asarray(other_bucket_sizes)
        self.other_mods_list = np.asarray(other_mods_list).T


        # the tables count the number of observed keys for each bucket
        self.tables = np.zeros((len(first_bucket_sizes), np.max(first_bucket_sizes)),dtype=int)

    def compute_keys(self, frames):
        """
        Assume that frames is a list of vectors, each representing a frame
        """
        # compute the signs of the dot products with the random vectors
        first_frame = frames[0]
        first_binaries = np.sign(np.asarray(first_frame).dot(self.projection_matrix))
        first_keys = np.cast['int'](first_binaries.dot(self.first_mods_list)) % self.first_bucket_sizes

        diffs = [
            np.abs(frames[i+1] - frames[i])
            for i in range(len(frames)-1)
        ]
        other_binaries =
        return first_keys

    def inc_keys(self, keys):
        """
        Increment hash table counts for many items (row-wise stacked as a matrix)
        """
        for idx in range(len(self.first_bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_keys(self, keys):
        """
        For each item, return the min of all counts from all buckets.
        """
        all_counts = []
        for idx in range(len(self.first_bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        counts = np.asarray(all_counts).min(axis=0)
        return counts

    def reset(self):
        self.tables = np.zeros(
            (len(self.first_bucket_sizes), np.max(self.first_bucket_sizes))
        )
