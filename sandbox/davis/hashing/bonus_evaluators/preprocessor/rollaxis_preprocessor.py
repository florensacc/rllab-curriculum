from sandbox.davis.hashing.bonus_evaluators.preprocessor.base import Preprocessor

import numpy as np


class RollaxisPreprocessor(Preprocessor):
    """
    Takes inputs of shape (N, W, H, C) as given by ALE and turns them into (N, C*W*H) as required
    by lasagne.
    """
    def process(self, items):
        return np.rollaxis(items, -1, 1).reshape(items.shape[0], -1)
