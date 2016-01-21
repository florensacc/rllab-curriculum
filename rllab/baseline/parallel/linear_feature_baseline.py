from rllab.baseline.base import Baseline
from rllab.misc.overrides import overrides
from rllab.sampler import parallel_sampler
import numpy as np


G = parallel_sampler.G


def worker_compute():
    baseline = G.baseline
    paths = G.paths
    featmat = np.concatenate([baseline.features(path) for path in paths])
    returns = np.concatenate([path["returns"] for path in paths])
    ATA = featmat.T.dot(featmat)
    ATb = featmat.T.dot(returns)
    return ATA, ATb


class LinearFeatureBaseline(Baseline):

    def __init__(self, mdp):
        self.coeffs = None

    @property
    @overrides
    def algorithm_parallelized(self):
        return True

    @overrides
    def get_param_values(self, **tags):
        return self.coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self.coeffs = val

    def features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, al**3], axis=1)

    @overrides
    def fit(self):
        mats = parallel_sampler.run_map(worker_compute)
        ATA = sum(mat[0] for mat in mats)
        ATb = sum(mat[1] for mat in mats)
        self.coeffs = np.linalg.solve(ATA + 1e-5*np.eye(ATA.shape[0]), ATb)

    @overrides
    def predict(self, path):
        if self.coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self.features(path).dot(self.coeffs)
