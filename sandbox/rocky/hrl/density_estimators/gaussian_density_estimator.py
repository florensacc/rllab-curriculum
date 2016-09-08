

import numpy as np
import theano.tensor as TT
from .base import DensityEstimator


class GaussianDenstiyEstimator(DensityEstimator):
    def __init__(self, data_dim, covariance_type='diag', reg_coeff=1e-5, name=None):
        self.data_dim = data_dim
        self.mean = np.zeros((data_dim,))
        self.cov = np.eye(data_dim)
        self.covariance_type = covariance_type
        self.reg_coeff = reg_coeff
        self._update_stats()

    def _update_stats(self):
        self.log_det_cov = np.linalg.slogdet(self.cov)[1]
        self.L = np.linalg.cholesky(self.cov)
        self.invL = np.linalg.inv(self.L)

    def fit(self, xs):
        xs = np.asarray(xs)
        self.mean = np.mean(xs, axis=0)
        if self.covariance_type == 'diag':
            self.cov = np.diag(np.var(xs, axis=0)) + np.eye(self.data_dim) * self.reg_coeff
        else:
            self.cov = np.cov(xs.T) + np.eye(self.data_dim) * self.reg_coeff
        self._update_stats()

    def predict_log_likelihood(self, xs):
        const = - 0.5 * np.log(2 * np.pi) * self.data_dim - 0.5 * self.log_det_cov
        return const - 0.5 * np.sum(np.square((xs - self.mean).dot(self.invL.T)), axis=1)

    def log_likelihood_sym(self, x_var):
        raise NotImplementedError("need repair!")
        const = - 0.5 * np.log(2 * np.pi) * self.data_dim - 0.5 * self.log_det_cov
        return const - 0.5 * TT.sum(TT.square((x_var - self.mean).dot(self.invL.T)), axis=1)
