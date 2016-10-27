from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.nary_hash import NaryHash
import numpy as np

class BassHash(NaryHash):
    """
    Use the discretized bass feature for hashing
    """
    def __init__(self,
            bass,
            n_channel,
            img_width,
            img_height,
            bucket_sizes=None,
            parallel=False,
        ):
        self.image_shape = (img_height, img_width, n_channel)
        self.item_dim = None
        self.bass = bass
        dim_key = self.bass.get_feature_length(self.image_shape)

        super().__init__(
            n=bass.n_bin,
            dim_key=dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )


    def compute_nary_keys(self, rgb_images):
        naries = np.asarray([self.bass.compute_feature_nary(I) for I in rgb_images])
        return naries
