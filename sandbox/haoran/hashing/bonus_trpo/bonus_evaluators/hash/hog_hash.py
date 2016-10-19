from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.binary_hash import BinaryHash
import numpy as np

class HOGHash(BinaryHash):
    """
    Binarize hog features by thresholding the relative weights of oriented gradients in each cell
    (optionally append a SimHash to it?)
    """
    def __init__(self,
            hog,
            n_channel,
            img_width,
            img_height,
            threshold=0.3,
            extract_channel_wise=False,
            bucket_sizes=None,
            parallel=False,
        ):

        self.hog = hog
        self.n_channel = n_channel
        self.img_width = img_width
        self.img_height = img_height
        self.item_dim = np.prod([n_channel,img_width,img_height])
        self.threshold = threshold
        self.extract_channel_wise = extract_channel_wise
        self.feature_shape = self.hog.get_feature_shape(
            image_shape=(n_channel,img_width,img_height)
        )
        if self.extract_channel_wise:
            dim_key = np.prod(self.feature_shape) * n_channel
        else:
            dim_key = np.prod(self.feature_shape)

        super().__init__(
            dim_key=dim_key,
            bucket_sizes=bucket_sizes,
            parallel=parallel,
        )

    def __getstate__(self):
        return super().__getstate__()

    def compute_binary_keys(self, items):
        hogs = self.compute_hogs(items)
        binaries = []
        for hog in hogs:
            binaries.append(np.sign(hog.ravel() - self.threshold))
        binaries = np.asarray(binaries)
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
