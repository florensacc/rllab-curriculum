import numpy as np


class ZeroMIEvaluator(object):
    def __init__(self, env_spec, policy):
        pass

    def predict(self, path):
        return np.zeros_like(path["rewards"])

    def log_diagnostics(self, paths):
        pass

    def fit(self, paths):
        pass

