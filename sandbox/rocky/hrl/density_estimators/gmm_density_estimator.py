from __future__ import print_function
from __future__ import absolute_import
from sklearn.mixture import GMM, DPGMM, VBGMM
from .base import DensityEstimator
import numpy as np
import theano
import theano.tensor as TT

floatX = theano.config.floatX


def _log_multivariate_normal_density_diag_sym(x_var, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""
    n_samples, n_dim = x_var.shape
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + TT.sum(TT.log(covars), 1)
                  + TT.sum((means ** 2) / covars, 1)
                  - 2 * TT.dot(x_var, (means / covars).T)
                  + TT.dot(x_var ** 2, (1.0 / covars).T))
    return lpr


def _log_sum_exp(x, axis=None, keepdims=True):
    ''' Numerically stable theano version of the Log-Sum-Exp trick'''
    x_max = TT.max(x, axis=axis, keepdims=True)

    preres = TT.log(TT.sum(TT.exp(x - x_max), axis=axis, keepdims=keepdims))
    return preres + x_max.reshape(preres.shape)


class GMMDensityEstimator(DensityEstimator):
    def __init__(self, data_dim, n_components=20, covariance_type='diag', reg_coeff=1e-5, name=None):
        self.data_dim = data_dim
        self.n_components = n_components
        self.reg_coeff = reg_coeff
        self.gmm = GMM(n_components=n_components, covariance_type=covariance_type)
        self.means_var = theano.shared(np.zeros((n_components, data_dim), dtype=floatX), name="means")
        self.covars_var = theano.shared(np.ones((n_components, data_dim), dtype=floatX), name="covars")
        self.weights_var = theano.shared(np.ones((n_components,), dtype=floatX) * 1.0 / n_components, name="weights")

    def fit(self, xs):
        self.gmm.fit(xs)
        self.means_var.set_value(np.cast[floatX](self.gmm.means_))
        self.covars_var.set_value(np.cast[floatX](self.gmm.covars_))
        self.weights_var.set_value(np.cast[floatX](self.gmm.weights_))

    def predict_log_likelihood(self, xs):
        return self.gmm.score_samples(xs)[0]

    def log_likelihood_sym(self, x_var):
        assert self.gmm.covariance_type == 'diag'
        lpr = _log_multivariate_normal_density_diag_sym(x_var, self.means_var, self.covars_var) + TT.log(
            self.weights_var)
        return _log_sum_exp(lpr, axis=1, keepdims=False)
