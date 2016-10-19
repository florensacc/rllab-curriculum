import numpy as np
from sandbox.davis.hashing.bonus_evaluators.hashing_bonus_evaluator import HashingBonusEvaluator


class BuiltinHashBonusEvaluator(HashingBonusEvaluator):
    def __init__(self, env_spec, granularity=None, bucket_sizes=[999931]):
        self.bucket_sizes = bucket_sizes
        self.granularity = granularity
        self.tables = np.zeros((1, bucket_sizes[0]))

    def compute_keys(self, observations):
        if self.granularity is not None:
            observations = self.discretize(observations)
        hashes = [hash(tuple(obs)) % self.bucket_sizes[0] for obs in observations]
        return np.array(hashes).reshape(-1, 1)

    def discretize(self, observations):
        return np.around(observations / self.granularity)
