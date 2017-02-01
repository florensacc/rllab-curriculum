import numpy as np
import itertools
from rllab.misc import logger
from sandbox.carlos_snn.bonus_evaluators.base import BonusEvaluator


class HashingBonusEvaluator(BonusEvaluator):
    def __init__(self, env_spec, dim_key=128, bucket_sizes=None):
        # Hashing function: SimHash
        if bucket_sizes is None:
            # some large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]
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
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        obs_dim = env_spec.observation_space.flat_dim
        self.projection_matrix = np.random.normal(size=(obs_dim, dim_key))

    def compute_keys(self, observations):
        binaries = np.sign(np.asarray(observations).dot(self.projection_matrix))
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, observations):
        keys = self.compute_keys(observations)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def query_hash(self, observations):
        keys = self.compute_keys(observations)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        counts = np.asarray(all_counts).min(axis=0)
        return counts

    def fit_before_process_samples(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        before_counts = self.query_hash(observations)
        self.inc_hash(observations)
        counts = self.query_hash(observations)
        logger.record_tabular_misc_stat("StateCount", counts)
        new_state_count = len(np.where(before_counts == 0)[0])
        logger.record_tabular('NewStateCount', new_state_count)

    def predict(self, path):
        counts = self.query_hash(path["observations"])
        return 1. / np.maximum(1., np.sqrt(counts))

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        total_hash_bonus = np.sum([np.sum(self.predict(path)) for path in paths])
        avg_hash_bonus = np.mean([np.sum(self.predict(path)) for path in paths])
        logger.record_tabular('Total_HashBonus', total_hash_bonus)
        logger.record_tabular('AvgPath_HashBonus', avg_hash_bonus)

