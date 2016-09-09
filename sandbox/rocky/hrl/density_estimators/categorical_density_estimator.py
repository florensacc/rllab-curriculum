

import numpy as np
import theano
import theano.tensor as TT
from .base import DensityEstimator


class CategoricalDenstiyEstimator(DensityEstimator):
    def __init__(self, data_dim, name=None):
        self.data_dim = data_dim
        self.prob_var = theano.shared(value=np.ones((data_dim,)) * 1.0 / data_dim, name="prob")

    def fit(self, xs):
        self.prob_var.set_value(np.mean(xs, axis=0))

    def predict_log_likelihood(self, xs):
        probs = self.prob_var.get_value()
        N = xs.shape[0]
        x_prob = np.sum(np.tile(probs.reshape((1, -1)), (N, 1)) * xs, axis=1)
        return np.log(x_prob + 1e-8)

    def log_likelihood_sym(self, x_var):
        N = x_var.shape[0]
        x_prob = TT.sum(TT.tile(self.prob_var.reshape((1, -1)), (N, 1)) * x_var, axis=1)
        return TT.log(x_prob + 1e-8)
