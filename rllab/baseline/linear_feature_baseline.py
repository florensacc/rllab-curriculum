from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np

class LinearFeatureBaseline(Baseline):

    def __init__(self, mdp):
        self.coeffs = None

    @overrides
    def get_param_values(self):
        return self.coeffs

    @overrides
    def set_param_values(self, val):
        self.coeffs = val

    def _features(self, path):
        o = np.clip(path["observations"], -10,10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1,1)/100.0
        return np.concatenate([o, o**2, al, al**2, al**3, np.ones((l,1))], axis=1)

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
