from sandbox.rocky.tf.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np


class LinearFeatureBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5, prediction_momentum=0.9):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self._prediction_momentum = prediction_momentum

    def get_param_values(self, **tags):
        return self._coeffs

    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        obs = np.concatenate([path["observations"], [path["last_observation"]]], axis=0)
        o = np.clip(obs, -10, 10)
        l = len(path["rewards"]) + 1
        al = (path["start_t"] + np.arange(l)).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        prev_predictions = np.concatenate([path["baselines"] for path in paths])
        smoothed_target = returns * (1 - self._prediction_momentum) + prev_predictions * self._prediction_momentum
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(smoothed_target)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]) + 1)
        return self._features(path).dot(self._coeffs)
