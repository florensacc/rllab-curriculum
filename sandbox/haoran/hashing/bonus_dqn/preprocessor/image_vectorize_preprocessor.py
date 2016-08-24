from sandbox.haoran.hashing.preprocessor.base import Preprocessor

class ImageVectorizePreprocessor(Preprocessor):
    def __init__(self,n_chanllel,width,height):
        self._input_dim = (n_chanllel,width,height)
        self._output_dim = n_chanllel * width * height

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def process(self,imgs):
        """
        Assume that imgs have shape (batch_size, n_chanllel, width, height)
        """
        batch_size = imgs.shape[0]
        return imgs.reshape((batch_size, self._output_dim))
