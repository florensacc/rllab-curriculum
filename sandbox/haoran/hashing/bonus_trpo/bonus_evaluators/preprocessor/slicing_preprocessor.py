from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.base import Preprocessor
import numpy as np

class SlicingPreprocessor(Preprocessor):
    def __init__(self, input_dim, start, stop, step):
        self._input_dim = input_dim
        assert start >= 0 and start < input_dim and stop >= 0 and stop <= input_dim and step > 0
        self._start = start
        self._stop = stop
        self._step = step
        self._output_dim = (stop - start) // step

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def process(self,inputs):
        """
        Assume inputs have shape (batch_size, input_dim)
        """
        assert len(inputs.shape) == 2 and inputs.shape[1] == self._input_dim

        return inputs[:,self._start : self._stop : self._step]
