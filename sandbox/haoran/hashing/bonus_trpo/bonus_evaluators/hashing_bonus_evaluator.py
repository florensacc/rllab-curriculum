from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import itertools
from rllab.misc import logger


class HashingBonusEvaluator(object):
    def __init__(self, env_spec, dim_key=128, bucket_sizes=None,bonus_form="1/sqrt(n)"):
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
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        obs_dim = env_spec.observation_space.flat_dim
        self.projection_matrix = np.random.normal(size=(obs_dim, dim_key))
        self.total_state_count = 0
        self.bonus_form = bonus_form


    def compute_keys(self, observations):
        observations = np.cast['int']((observations + 1) * 0.5 * 255.0)
        binaries = np.sign(np.asarray(observations).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, observations):
        keys = self.compute_keys(observations)
        for idx in xrange(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_hash(self, observations):
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
        logger.record_tabular_misc_stat('StateCount',counts)
        new_state_count = len(np.where(counts==1)[0])
        logger.record_tabular('NewSteateCount',new_state_count)

        self.total_state_count += new_state_count
        logger.record_tabular('TotalStateCount',self.total_state_count)

    def predict(self, path):
        counts = np.maximum(1.,self.query_hash(path["observations"]))
        if self.bonus_form == "1/n":
            bonuses = 1./counts
        elif self.bonus_form == "1/sqrt(n)":
            bonuses = 1./np.sqrt(counts)
        elif self.bonus_form == "1/log(n+1)":
            bonuses = 1./np.log(counts + 1)
        else:
            raise NotImplementedError
        return bonuses

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        pass
