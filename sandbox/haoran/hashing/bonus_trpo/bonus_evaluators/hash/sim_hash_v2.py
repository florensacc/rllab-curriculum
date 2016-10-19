from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.binary_hash import BinaryHash
from rllab.misc.overrides import overrides
import numpy as np

class SimHashV2(BinaryHash):
    """
    Same as SimHash, but defined as a subclass of BinaryHash
    """
    def __init__(self,
            item_dim,
            dim_key=128,
            bucket_sizes=None,
            parallel=False,
        ):
        super().__init__(
            dim_key=dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )

        # each column is a vector of uniformly random orientation
        self.projection_matrix = np.random.normal(size=(item_dim, dim_key))
        self.item_dim = item_dim

        self.snapshot_list.append("projection_matrix")

    def __getstate__(self):
        return super().__getstate__()

    def compute_binary_keys(self, items):
        binaries = np.sign(np.asarray(items).dot(self.projection_matrix))
        return binaries
