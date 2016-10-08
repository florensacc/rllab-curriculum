from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.preprocessor.base import Preprocessor
import numpy as np

class IdentityPreprocessor(Preprocessor):
    def __init__(self,input_dim):
        self._input_dim = input_dim
        self._output_dim = input_dim

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def process(self,items):
        return items
