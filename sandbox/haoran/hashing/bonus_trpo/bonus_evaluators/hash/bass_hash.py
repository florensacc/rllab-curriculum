from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.nary_hash import NaryHash
import numpy as np

class BassHash(NaryHash):
    """
    Use the discretized bass feature for hashing
    """
    def __init__(self,
            bass,
            bucket_sizes=None,
            parallel=False,
        ):
        self.item_dim = None
        self.bass = bass
        dim_key = self.bass.get_feature_length()

        super().__init__(
            n=bass.n_bin,
            dim_key=dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )


    def compute_nary_keys(self, rgb_images):
        naries = self.bass.compute_features_nary(rgb_images)
        return naries
