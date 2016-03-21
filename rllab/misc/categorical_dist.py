import theano.tensor as TT
import numpy as np


def from_onehot_sym(x_var):
    return TT.nonzero(x_var)[1]


def kl_sym(old_prob_var, new_prob_var):
    """
    Compute the symbolic KL divergence of two categorical distributions
    """
    if old_prob_var.ndim == 2:
        # Assume layout is N * A
        return TT.sum(
            old_prob_var * (TT.log(old_prob_var + 1e-8) - TT.log(new_prob_var + 1e-8)),
            axis=1
        )
    elif old_prob_var.ndim == 3:
        # Assume layout is N * T * A
        return TT.sum(
            old_prob_var * (TT.log(old_prob_var + 1e-8) - TT.log(new_prob_var + 1e-8)),
            axis=2
        )
    else:
        raise NotImplementedError


def kl(old_prob, new_prob):
    """
    Compute the KL divergence of two categorical distributions
    """
    if old_prob.ndim == 2:
        return np.sum(
            old_prob * (np.log(old_prob) - np.log(new_prob)),
            axis=1
        )
    elif old_prob.ndim == 3:
        return np.sum(
            old_prob * (np.log(old_prob) - np.log(new_prob)),
            axis=2
        )
    else:
        raise NotImplementedError

def likelihood_ratio_sym(x_var, old_prob_var, new_prob_var):
    if old_prob_var.ndim == 2:
        # Assume layout is N * A
        N = old_prob_var.shape[0]
        x_inds = from_onehot_sym(x_var)
        return new_prob_var[TT.arange(N), x_inds] / old_prob_var[TT.arange(N), x_inds]
    elif old_prob_var.ndim == 3:
        # Assume layout is N * T * A
        a_dim = x_var.shape[-1]
        flat_ratios = likelihood_ratio_sym(
            x_var.reshape((-1, a_dim)),
            old_prob_var.reshape((-1, a_dim)),
            new_prob_var.reshape((-1, a_dim))
        )
        return flat_ratios.reshape(old_prob_var.shape[:2])
    else:
        raise NotImplementedError


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=1)


def log_prob_sym(xs, probs):
    if probs.ndim == 2:
        # Assume layout is N * A
        N = probs.shape[0]
        return TT.log(probs[TT.arange(N), from_onehot_sym(xs)])
    elif probs.ndim == 3:
        # Assume layout is N * T * A
        a_dim = xs.shape[-1]
        flat_log_prob = log_prob_sym(xs.reshape((-1, a_dim)), probs.reshape((-1, a_dim)))
        return flat_log_prob.reshape(probs.shape[:2])
    else:
        raise NotImplementedError
