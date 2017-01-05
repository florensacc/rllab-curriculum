from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.nary_hash import NaryHash
from rllab.misc.overrides import overrides
import numpy as np

class SimHashV3(NaryHash):
    """
    Same as SimHash, but defined as a subclass of NaryHash
    """
    def __init__(self,
            item_dim,
            dim_key=128,
            bucket_sizes=None,
            parallel=False,
            standard_code=False,
        ):
        self.standard_code = standard_code
        super().__init__(
            n=2,
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

    def compute_nary_keys(self, items):
        binaries = np.sign(np.asarray(items).dot(self.projection_matrix))
        if self.standard_code:
            binaries = (0.5*(binaries+1)).astype(int)
        return binaries

    def get_copy(self):
        h = SimHashV3(
            item_dim=self.item_dim,
            dim_key=self.dim_key,
            bucket_sizes=self.bucket_sizes,
            parallel=self.parallel,
            standard_code=self.standard_code,
        )
        h.projection_matrix = np.copy(self.projection_matrix)
        return h
