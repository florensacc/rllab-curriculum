import theano.tensor as TT
import numpy as np


def kl_sym(old_means, old_log_stds, new_means, new_log_stds):
    """
    Compute the KL divergence of two multivariate Gaussian distribution with
    diagonal covariance matrices
    """
    old_std = TT.exp(old_log_stds)
    new_std = TT.exp(new_log_stds)
    # means: (N*A)
    # std: (N*A)
    # formula:
    # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
    # ln(\sigma_2/\sigma_1)
    numerator = TT.square(old_means - new_means) + \
        TT.square(old_std) - TT.square(new_std)
    denominator = 2 * TT.square(new_std) + 1e-8
    return TT.sum(
        numerator / denominator + new_log_stds - old_log_stds, axis=1)


def log_normal_pdf(xs, means, log_stds):
    normalized = (xs - means) / TT.exp(log_stds)
    return -0.5 * TT.square(normalized) - np.log((2 * np.pi)**0.5) - log_stds


def likelihood_ratio_sym(xs, old_means, old_log_stds, new_means, new_log_stds):
    logli_new = log_normal_pdf(xs, new_means, new_log_stds)
    logli_old = log_normal_pdf(xs, old_means, old_log_stds)
    return TT.exp(TT.sum(logli_new - logli_old, axis=1))


def log_likelihood_sym(xs, means, log_stds):
    zs = (xs - means) / TT.exp(log_stds)
    return - TT.sum(log_stds, axis=1) - \
        0.5 * TT.sum(TT.square(zs), axis=1) - \
        0.5 * means.shape[1] * np.log(2 * np.pi)


def entropy(means, log_stds):
    return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=1)
