from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.binary_hash import BinaryHash
import numpy as np


class FeatureSimHash(BinaryHash):
    """
    SimHash on output of a feature extractor
    """
    def __init__(
        self,
        feature_extractor,
        dim_key=128,
        bucket_sizes=None,
        parallel=False,
    ):
        self.feature_extractor = feature_extractor

        super().__init__(
            dim_key=dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )

        # each column is a vector of uniformly random orientation
        self.item_dim = None  # ???
        self.projection_matrix = np.random.normal(size=(feature_extractor.feature_dim, dim_key))

        self.snapshot_list.append("projection_matrix")

    def __getstate__(self):
        return super().__getstate__()

    def compute_binary_keys(self, items):
        features = self.feature_extractor.get_features(items)
        binaries = np.sign(np.asarray(features).dot(self.projection_matrix))
        return binaries
