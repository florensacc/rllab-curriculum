import numpy as np
from sandbox.rocky.chainer.distributions.base import Distribution
import chainer.functions as F


class DiagonalGaussian(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        old_means = old_dist_info["mean"]
        old_log_stds = old_dist_info["log_std"]
        new_means = new_dist_info["mean"]
        new_log_stds = new_dist_info["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = np.square(old_means - new_means) + \
                    np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)
        # more lossy version
        # return TT.sum(
        #     numerator / denominator + TT.log(new_std) - TT.log(old_std ), axis=-1)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["log_std"]
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices
        """
        old_std = F.exp(old_log_stds)
        new_std = F.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = F.square(old_means - new_means) + \
                    F.square(old_std) - F.square(new_std)
        denominator = 2 * F.square(new_std) + 1e-8
        return F.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return F.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        x_var = F.cast(x_var, np.float32)
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["log_std"]
        zs = (x_var - means) / F.exp(log_stds)
        return - F.sum(log_stds, axis=-1) - \
               0.5 * F.sum(F.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def sample(self, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def entropy(self, dist_info):
        log_stds = dist_info["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def entropy_sym(self, dist_info):
        log_stds = dist_info["log_std"]
        return F.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_info_specs(self):
        return [("mean", (self.dim,)), ("log_std", (self.dim,))]
