import numpy as np


class ZeroBonusEvaluator(object):

    def fit(self, paths):
        pass

    def predict(self, path):
        return np.zeros_like(path["rewards"])