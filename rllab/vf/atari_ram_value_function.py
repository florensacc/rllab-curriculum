from rllab.vf.base import ValueFunction
from rllab.misc.overrides import overrides
import numpy as np


class AtariRamLinearValueFunction(ValueFunction):

    def __init__(self):
        self.coeffs = None

    @overrides
    def get_param_values(self):
        return self.coeffs

    @overrides
    def set_param_values(self, val):
        self.coeffs = val

    def _features(self, path):
        o = path["observations"]
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1,1) / 50.0
        return np.concatenate([o, al, al**2, np.ones((l,1))], axis=1)

    @overrides
    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self.coeffs = np.linalg.lstsq(featmat, returns)[0]

    @overrides
    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self.coeffs)
