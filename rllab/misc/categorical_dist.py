import theano.tensor as TT
import numpy as np


def from_onehot_sym(x_var):
    return TT.nonzero(x_var)[1]


def kl_sym(old_prob_var, new_prob_var):
    """
    Compute the symbolic KL divergence of two categorical distributions
    """
    return TT.sum(
        old_prob_var * (TT.log(old_prob_var) - TT.log(new_prob_var)),
        axis=1
    )


def kl(old_prob, new_prob):
    """
    Compute the KL divergence of two categorical distributions
    """
    return np.sum(
        old_prob * (np.log(old_prob) - np.log(new_prob)),
        axis=1
    )


def likelihood_ratio_sym(x_var, old_prob_var, new_prob_var):
    N = old_prob_var.shape[0]
    x_inds = from_onehot_sym(x_var)
    return new_prob_var[TT.arange(N), x_inds] / old_prob_var[TT.arange(N), x_inds]


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=1)


def log_prob_sym(xs, probs):
    N = probs.shape[0]
    return TT.log(probs[TT.arange(N), from_onehot_sym(xs)])
