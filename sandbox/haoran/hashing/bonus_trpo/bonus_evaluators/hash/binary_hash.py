from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
import numpy as np
import multiprocessing as mp
import copy

class BinaryHash(Hash):
    def __init__(self,dim_key, bucket_sizes=None,parallel=False):
        """
        Encode each item (vector) a binary code (sequence of 1 or -1) of length dim_key. Then further compress the binary code as \sum_{i=0}^{dim_key} b_i 2^i (mod bucket_size) for each bucket. The bucket sizes are primes, so that we obtain distinct keys. Bucket keys are redundant to reduce the error caused by compression (see the query part).
        A key is finally represented as indices in the bukckets.
        When a new item is given, increment the key counts for all buckets.
        When querying the count of an item, compute the key counts for all buckets but return the minimum. This way the count is less prone to errors caused by key compression.
        """
        # each column is a vector of uniformly random orientation
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
        self.parallel = parallel
        if parallel:
            self.tables_lock = mp.Value('i')
            self.tables = np.frombuffer(
                mp.RawArray('i', int(len(bucket_sizes) * np.max(bucket_sizes))),
                np.int32,
            )
            self.tables = self.tables.reshape((len(bucket_sizes), np.max(bucket_sizes)))
            self.unpicklable_list = ["tables_lock","tables"]
            self.snapshot_list = ["tables"]
        else:
            self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)),dtype=int)
            self.unpicklable_list = []
            self.snapshot_list = []

    def __getstate__(self):
        """ Do not pickle parallel objects. """
        state = dict()
        for k,v in iter(self.__dict__.items()):
            if k not in self.unpicklable_list:
                state[k] = v
            elif k in self.snapshot_list:
                state[k] = copy.deepcopy(v)
        return state

    def init_rank(self,rank):
        self.rank = rank

    def compute_binary_keys(self, items):
        """ to be implemented by subclasses """
        raise NotImplementedError

    def compute_keys(self, items):
        """
        Compute the keys for many items (row-wise stacked as a matrix)
        """
        # compute the signs of the dot products with the random vectors
        binaries = self.compute_binary_keys(items)
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_keys(self, keys):
        """
        Increment hash table counts for many items (row-wise stacked as a matrix)
        """
        if self.parallel:
            print("%d: before table lock"%(self.rank))
            with self.tables_lock.get_lock():
                print("%d: inside table lock"%(self.rank))
                for idx in range(len(self.bucket_sizes)):
                    np.add.at(self.tables[idx], keys[:, idx], 1)
            print("%d: exit table lock"%(self.rank))
        else:
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
        if self.parallel:
            with self.tables_lock.get_lock():
                self.tables = np.zeros(
                    (len(self.bucket_sizes), np.max(self.bucket_sizes))
                )
        else:
            self.tables = np.zeros(
                (len(self.bucket_sizes), np.max(self.bucket_sizes))
            )
