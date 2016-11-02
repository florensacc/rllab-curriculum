from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
import numpy as np
import multiprocessing as mp
import copy

class NaryHash(Hash):
    def __init__(self,n,dim_key, bucket_sizes=None,parallel=False):
        """
        Simple extension of BinaryHash to n-ary keys

        :param bucket_sizes: None means implementing the hash table with python dictionary
        """
        # each column is a vector of uniformly random orientation
        self.n = n
        self.dim_key = dim_key
        self.counter = "tables" if bucket_sizes is not None else "dict"

        self.parallel = parallel
        if self.counter == "tables":
            # precompute modulos of powers of 2
            mods_list = []
            for bucket_size in bucket_sizes:
                mod = 1
                mods = []
                for _ in range(dim_key):
                    mods.append(mod)
                    mod = (mod * n) % bucket_size
                mods_list.append(mods)
            self.bucket_sizes = np.asarray(bucket_sizes)
            self.mods_list = np.asarray(mods_list).T

            # the tables count the number of observed keys for each bucket
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
        else:
            if parallel:
                self.shared_dict = dict() # for force_compiling
                self.shared_dict_prefix = 'nary_hash_counter: '
                self.unpicklable_list = ["shared_dict"]
                self.snapshot_list = []
                #FIXME: snapshot only the hash counts rather than the entire shared dict
            else:
                self.counter_dict = dict()


    def __getstate__(self):
        """ Do not pickle parallel objects. """
        state = dict()
        for k,v in iter(self.__dict__.items()):
            if k not in self.unpicklable_list:
                state[k] = v
            elif k in self.snapshot_list:
                state[k] = copy.deepcopy(v)
        if self.parallel and self.counter == "dict":
            state["counter_dict"] = {
                key: value for key,value in self.shared_dict.items()
                if self.shared_dict_prefix in key
            }
        return state

    def init_rank(self,rank):
        assert self.parallel
        self.rank = rank

    def init_shared_dict(self, shared_dict):
        assert self.parallel
        self.shared_dict = shared_dict

    def compute_nary_keys(self, items):
        """ to be implemented by subclasses """
        raise NotImplementedError

    def compute_keys(self, items):
        """
        Compute the keys for many items (row-wise stacked as a matrix)
        """
        naries = self.compute_nary_keys(items)
        if self.counter == "tables":
            # compute the signs of the dot products with the random vectors
            keys = np.cast['int'](naries.dot(self.mods_list)) % self.bucket_sizes
        else:
            if self.parallel:
                def nary_to_string(nary):
                    return self.shared_dict_prefix + str(list(nary)).replace(' ','')
            else:
                def nary_to_string(nary):
                    return str(list(nary)).replace(' ','')
            keys = [nary_to_string(nary) for nary in naries]
        return keys

    def inc_keys(self, keys):
        """
        Increment hash table counts for many items (row-wise stacked as a matrix)
        """
        # w/ hash tables: store all keys in the shared dict; then update using one process; this way we are able to log exactly how many states are new
        if self.counter == "tables":
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
        else:
            if self.parallel:
                counter_dict = self.shared_dict
            else:
                counter_dict = self.counter_dict
            # multiprocessing makes sure the following operations are synchronized
            for key in keys:
                if key in counter_dict:
                    counter_dict[key] += 1
                else:
                    counter_dict[key] = 1


    def query_keys(self, keys):
        """
        For each item, return the min of all counts from all buckets.
        """
        if self.counter == "tables":
            all_counts = []
            for idx in range(len(self.bucket_sizes)):
                all_counts.append(self.tables[idx, keys[:, idx]])
            counts = np.asarray(all_counts).min(axis=0)
        else:
            counts = []
            counter_dict = self.shared_dict if self.parallel else self.counter_dict
            for key in keys:
                if key in counter_dict:
                    counts.append(counter_dict[key])
                else:
                    counts.append(0)
        return counts

    def reset(self):
        if self.counter == "tables":
            if self.parallel:
                with self.tables_lock.get_lock():
                    self.tables = np.zeros(
                        (len(self.bucket_sizes), np.max(self.bucket_sizes))
                    )
            else:
                self.tables = np.zeros(
                    (len(self.bucket_sizes), np.max(self.bucket_sizes))
                )
        else:
            # should delete all keys from shared_dict starting with prefix "nary_hash_counter"
            if self.parallel:
                for key in self.shared_dict.keys():
                    if self.shared_dict_prefix in key:
                        self.shared_dict.pop(key)
            else:
                self.counter_dict = dict()


    def total_state_count(self):
        """
        This is not a precise count of the total number of states visited.
        Count the number of non-zero entries in each bucket; then return the maximum of them.
        """
        if self.counter == "tables":
            return np.max([np.count_nonzero(T) for T in self.tables])
        else:
            if self.parallel:
                all_keys = [
                    key for key in self.shared_dict.keys()
                    if self.shared_dict_prefix in key
                ]
                return len(all_keys)
            else:
                return len(self.counter_dict.keys())
