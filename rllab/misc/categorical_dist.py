import theano.tensor as TT
import numpy as np


def from_onehot_sym(x_var):
    return TT.nonzero(x_var)[1]


def from_onehot(x_var):
    return np.nonzero(x_var)[1]


def kl_sym(old_info_vars, new_info_vars):
    """
    Compute the symbolic KL divergence of two categorical distributions
    """
    old_log_prob_var = old_info_vars["log_prob"]
    new_log_prob_var = new_info_vars["log_prob"]
    if old_log_prob_var.ndim == 2:
        # Assume layout is N * A
        return TT.sum(
            TT.exp(old_log_prob_var) * (old_log_prob_var - new_log_prob_var),
            axis=1
        )
    elif old_log_prob_var.ndim == 3:
        # Assume layout is N * T * A
        return TT.sum(
            TT.exp(old_log_prob_var) * (old_log_prob_var - new_log_prob_var),
            axis=2
        )
    else:
        raise NotImplementedError


def kl(old_info, new_info):
    """
    Compute the KL divergence of two categorical distributions
    """
    old_log_prob = old_info["log_prob"]
    new_log_prob = new_info["log_prob"]
    if old_log_prob.ndim == 2:
        return np.sum(
            np.exp(old_log_prob) * (old_log_prob - new_log_prob),
            axis=1
        )
    elif old_log_prob.ndim == 3:
        return np.sum(
            np.exp(old_log_prob) * (old_log_prob - new_log_prob),
            axis=2
        )
    else:
        raise NotImplementedError


def likelihood_ratio_sym(x_var, old_info_vars, new_info_vars):
    old_log_prob_var = old_info_vars["log_prob"]
    new_log_prob_var = new_info_vars["log_prob"]
    if old_log_prob_var.ndim == 2:
        # Assume layout is N * A
        N = old_log_prob_var.shape[0]
        x_inds = from_onehot_sym(x_var)
        return TT.exp(new_log_prob_var[TT.arange(N), x_inds] - old_log_prob_var[TT.arange(N), x_inds])
    elif old_log_prob_var.ndim == 3:
        # Assume layout is N * T * A
        a_dim = x_var.shape[-1]
        flat_ratios = likelihood_ratio_sym(
            x_var.reshape((-1, a_dim)),
            old_log_prob_var.reshape((-1, a_dim)),
            new_log_prob_var.reshape((-1, a_dim))
        )
        return flat_ratios.reshape(old_log_prob_var.shape[:2])
    else:
        raise NotImplementedError


def entropy(info):
    log_probs = info["log_prob"]
    return -np.sum(np.exp(log_probs) * log_probs, axis=1)


def log_likelihood_sym(xs, info_vars):
    log_probs = info_vars["log_prob"]
    if log_probs.ndim == 2:
        # Assume layout is N * A
        xs = from_onehot_sym(xs)
        N = log_probs.shape[0]
        return log_probs[TT.arange(N), xs]
    elif log_probs.ndim == 3:
        # Assume layout is N * T * A
        a_dim = log_probs.shape[-1]
        flat_log_prob = log_likelihood_sym(xs.reshape((-1, a_dim)), log_probs.reshape((-1, a_dim)))
        return flat_log_prob.reshape(log_probs.shape[:2])
    else:
        raise NotImplementedError


def log_likelihood(xs, info):
    log_probs = info["log_prob"]
    if log_probs.ndim == 2:
        # Assume layout is N * A
        N = log_probs.shape[0]
        return log_probs[np.arange(N), from_onehot(xs)]
    elif log_probs.ndim == 3:
        # Assume layout is N * T * A
        a_dim = log_probs.shape[-1]
        flat_log_prob = log_likelihood_sym(xs.flatten(), log_probs.reshape((-1, a_dim)))
        return flat_log_prob.reshape(log_probs.shape[:2])
    else:
        raise NotImplementedError
