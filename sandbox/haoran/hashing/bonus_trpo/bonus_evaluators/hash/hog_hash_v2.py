from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.binary_hash import BinaryHash
import numpy as np

class HOGHashV2(BinaryHash):
    """
    Directly apply a second hash to the HOG feature vector. The second hash to preserve certain continuity.
    """
    def __init__(self,
            hog,
            n_channel,
            img_width,
            img_height,
            second_hash,
            extract_channel_wise=False,
            bucket_sizes=None,
            parallel=False,
        ):

        self.hog = hog
        self.n_channel = n_channel
        self.img_width = img_width
        self.img_height = img_height
        assert isinstance(second_hash, BinaryHash)
        self.second_hash = second_hash
        self.item_dim = np.prod([n_channel,img_width,img_height])
        self.extract_channel_wise = extract_channel_wise
        self.feature_shape = self.hog.get_feature_shape(
            image_shape=(n_channel,img_width,img_height)
        )
        if self.extract_channel_wise:
            second_hash_input_dim = np.prod(self.feature_shape) * n_channel
        else:
            second_hash_input_dim = np.prod(self.feature_shape)
        assert second_hash_input_dim == second_hash.item_dim

        super().__init__(
            dim_key=second_hash.dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )

    def __getstate__(self):
        return super().__getstate__()

    def compute_binary_keys(self, items):
        hogs = self.compute_hogs(items)
        hogs_flat = hogs.reshape((hogs.shape[0],-1))
        binaries = self.second_hash.compute_binary_keys(hogs_flat)
        return binaries

    def compute_hogs(self,items):
        if len(items.shape) != 4:
            assert len(items.shape) == 2
            # de-vectorize the images
            n_items = items.shape[0]
            items = items.reshape(
                (n_items,self.n_channel,self.img_width,self.img_height))

        batch_size, n_channel, width, height = items.shape
        if self.extract_channel_wise:
            n_bins, hog_width, hog_height = self.feature_shape
            hogs = np.zeros(
                (batch_size, n_channel, n_bins, hog_width, hog_height),
                dtype=float
            )
            for c in range(n_channel):
                hogs_c = self.hog.compute_hogs(
                    np.expand_dims(items[:,c,:,:],axis=1)
                )
                hogs[:,c,:,:,:] = hogs_c
        else:
            hogs = self.hog.compute_hogs(items)
        return hogs
