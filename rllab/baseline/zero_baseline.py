import numpy as np
from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides


class ZeroBaseline(Baseline):

    def __init__(self, mdp):
        pass

    @overrides
    def get_param_values(self):
        return None

    @overrides
    def set_param_values(self, val):
        pass

    @overrides
    def fit(self, paths):
        pass

    @overrides
    def predict(self, path):
        return np.zeros_like(path["returns"])
