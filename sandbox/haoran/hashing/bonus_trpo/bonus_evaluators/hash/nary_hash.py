from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.base import Hash
from rllab.misc import logger
import numpy as np
import multiprocessing as mp
import copy
import sys

class NaryHash(Hash):
    def __init__(self,n,dim_key, bucket_sizes=None,parallel=False):
        """
        Simple extension of BinaryHash to n-ary keys

        dictionary counter: convert an n-ary key into a uint64 tuple; every x consecutive n-ary digits are converted to a single uint64 number, where x = floor(log_n(2**64))

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
            self.digit_group_len = int(np.floor(64 * np.log(2) / np.log(self.n)))
            self.key_len = int(np.ceil(self.dim_key / self.digit_group_len))
            self.powers = [self.n ** j for j in range(self.digit_group_len)]
            if parallel:
                self.shared_dict = dict() # a placeholder for force_compiling; to be replaced by a true shared_dict pass down from the manager
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
                if self.validate_key(key)
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
        naries = self.compute_nary_keys(items).astype(np.uint64)
        if self.counter == "tables":
            # compute the signs of the dot products with the random vectors
            keys = np.cast['int'](naries.dot(self.mods_list)) % self.bucket_sizes
        else:
            # Old implementation: use bytes as keys
            # if self.n != 2:
            #     raise NotImplementedError
            # else:
            #     """
            #     group every 8 bits into an uint8, which transforms into a byte
            #     """
            #     def binary_to_bytes(binary):
            #         bytes_ints = []
            #         byte_str = ''
            #         for i,bit in enumerate(binary):
            #             if bit == 1:
            #                 byte_str = byte_str + '1'
            #             elif bit == -1 or bit == 0:
            #                 byte_str = byte_str + '0'
            #             else:
            #                 raise NotImplementedError
            #             bit_count = len(byte_str)
            #
            #             # group 8 bits
            #             if bit_count == 8:
            #                 bytes_ints.append(int(byte_str,2))
            #                 byte_str = ''
            #             # if not enough 8 bits left, pad zeros
            #             elif i == len(binary) - 1:
            #                 for j in range(8-bit_count):
            #                     byte_str = byte_str + '0'
            #                 bytes_ints.append(int(byte_str,2))
            #         return bytes(bytes_ints)
            #
            #     def binary_to_key(binary):
            #         if self.parallel:
            #             return self.shared_dict_prefix + binary_to_bytes(binary)
            #         else:
            #             return binary_to_bytes(binary)
            # keys = [binary_to_key(binary) for binary in naries]
            # input must have type int

            N,k = naries.shape # dimension
            assert k == self.dim_key
            keys = np.zeros((N, self.key_len),dtype=np.uint64)
            for i in range(0,k,self.digit_group_len):
                uints = np.zeros(N,dtype=np.uint64)
                for j in range(min(self.digit_group_len, k-i)):
                    uints = uints + naries[:,i+j] * self.powers[j]
                assert(uints.dtype == np.uint64)
                keys[:,i//self.digit_group_len] = uints
            keys = [tuple(key) for key in keys]
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
                    if self.validate_key(key):
                        self.shared_dict.pop(key)
            else:
                self.counter_dict = dict()

    def validate_key(self,key):
        """
        If the shared_dict has inserted by other objects, this function checks keys inserted by this object only. I am commenting it out as it is not necessary at the moment
        """
        # return isinstance(key, tuple) and len(key) = self.key_len and key[0].dtype==np.uint64
        return True

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
                    if self.validate_key(key)
                ]
                logger.log("Nary hash total state count: %d, estimated memory requirement: %.3f MB"%(len(all_keys), self.key_len * (64/8) * len(all_keys) / 1024**2))
                return len(all_keys)
            else:
                return len(self.counter_dict.keys())
