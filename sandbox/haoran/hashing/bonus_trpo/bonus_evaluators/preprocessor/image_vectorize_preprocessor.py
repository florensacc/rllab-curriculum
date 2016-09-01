from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.base import Preprocessor
import numpy as np

class ImageVectorizePreprocessor(Preprocessor):
    def __init__(self,n_channel,width,height,slices=[None,None,None]):
        self._input_dim = (n_channel,width,height)
        self.slices = []
        for s,size in zip(slices,[n_channel,width,height]):
            if s is None:
                s = slice(0,size,1)
            else:
                assert s.start >= 0 and s.stop <= size and s.step >= 1
            self.slices.append(s)

        self.sliced_dims = [
            (s.stop - s.start) // s.step
            for s in self.slices
        ]
        self._output_dim = np.prod(self.sliced_dims)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def process(self,imgs):
        """
        Assume that imgs have shape (batch_size, n_chanllel, width, height)
        """
        batch_size = imgs.shape[0]
        sliced_imgs = imgs[:,self.slices[0], self.slices[1], self.slices[2]]
        return sliced_imgs.reshape((batch_size, self._output_dim))
