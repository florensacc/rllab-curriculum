from __future__ import print_function
from __future__ import absolute_import

from nose2.tools import such
from sandbox.rocky.hrl.density_estimators.gaussian_density_estimator import GaussianDenstiyEstimator
from sandbox.rocky.hrl.density_estimators.gmm_density_estimator import GMMDensityEstimator
import numpy as np
from scipy.stats import multivariate_normal
import theano

with such.A("Gaussian Density Estimator") as it:
    @it.should
    def test_gaussian():
        est = GaussianDenstiyEstimator(data_dim=3)
        data = np.random.uniform(low=-1, high=1, size=(10, 3))
        est.fit(data)
        xs = np.random.uniform(low=-1, high=1, size=(2, 3))
        logli_ref = multivariate_normal.logpdf(xs, np.mean(data, axis=0), np.diag(np.var(data, axis=0)))
        logli = est.predict_log_likelihood(xs)
        np.testing.assert_allclose(logli_ref, logli, atol=1e-3)
        x_var = theano.shared(xs)
        logli_sym_val = est.log_likelihood_sym(x_var).eval()
        np.testing.assert_allclose(logli_ref, logli_sym_val, atol=1e-3)

it.createTests(globals())

with such.A("Mixture of Gaussian Density Estimator") as it:
    @it.should
    def test_gmm():
        est = GMMDensityEstimator(data_dim=3, n_components=5)
        mean1 = np.zeros((3,))
        cov1 = np.eye(3) * 0.01
        mean2 = np.ones((3,))
        cov2 = np.eye(3) * 0.01
        data = np.concatenate([
            np.random.multivariate_normal(mean=mean1, cov=cov1, size=100),
            np.random.multivariate_normal(mean=mean2, cov=cov2, size=100)
        ])
        est.fit(data)
        test_xs = np.random.multivariate_normal(mean=mean1, cov=cov1, size=5)
        logli_ref = est.predict_log_likelihood(test_xs)
        x_var = theano.shared(test_xs)
        logli_sym_val = est.log_likelihood_sym(x_var).eval()
        np.testing.assert_allclose(logli_ref, logli_sym_val)

it.createTests(globals())
