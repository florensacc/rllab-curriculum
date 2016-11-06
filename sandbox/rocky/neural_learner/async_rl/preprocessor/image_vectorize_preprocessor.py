from ..preprocessor.base import Preprocessor
from ..utils.shareable import Shareable
import numpy as np

class ImageVectorizePreprocessor(Preprocessor,Shareable):
    def __init__(self,n_channel,width,height,slices=[None,None,None],use_current_image=False):
        """
        Each slice is a tuple as (s.start, s.stop, s.step)
        """
        if use_current_image:
            slices = [(n_channel-1,n_channel,1),None,None]
        else:
            slices = [None,None,None]
        self.init_params = locals()
        self.init_params.pop("self")
        self._input_dim = (n_channel,width,height)
        self.slices = []
        self.sliced_dims = []
        for slice_args,size in zip(slices,[n_channel,width,height]):
            if slice_args is None:
                slice_args = (0,size,1)
            start,stop,step = slice_args
            assert start >= 0 and stop <= size and step >= 1
            self.slices.append(slice_args)
            self.sliced_dims.append((stop - start) / step)
        self._output_dim = np.prod(self.sliced_dims)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def output_dim(self):
        return self._output_dim

    def process(self,imgs):
        """
        Assume that imgs have shape (batch_size, n_chanllel, width, height)
        """
        batch_size = imgs.shape[0]
        slices = [
            slice(start,stop,step)
            for start,stop,step in self.slices
        ]
        sliced_imgs = imgs[:,slices[0], slices[1], slices[2]]
        return sliced_imgs.reshape((batch_size, self._output_dim))

    def process_copy(self):
        return ImageVectorizePreprocessor(**self.init_params)
