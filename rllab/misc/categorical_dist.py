import theano.tensor as TT
import numpy as np


def kl_sym(old_prob_var, new_prob_var):
    """
    Compute the KL divergence of two categorical distributions
    """
    return TT.sum(
        old_prob_var * (TT.log(old_prob_var) - TT.log(new_prob_var)),
        axis=1
    )


def likelihood_ratio_sym(x_var, old_prob_var, new_prob_var):
    N = old_prob_var.shape[0]
    return new_prob_var[TT.arange(N), x_var.reshape((-1,))] / \
        old_prob_var[TT.arange(N), x_var.reshape((-1,))]


def entropy(probs):
    return -np.sum(probs * np.log(probs), axis=1)


def log_prob_sym(xs, probs):
    N = probs.shape[0]
    return TT.log(probs[TT.arange(N), xs.reshape((-1,))])
