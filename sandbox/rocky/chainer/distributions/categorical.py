import numpy as np
from .base import Distribution
import chainer
import chainer.functions as F
from rllab.misc import special

TINY = 1e-8


def from_onehot(x_var):
    ret = np.zeros((len(x_var),), 'int32')
    nonzero_n, nonzero_a = np.nonzero(x_var)
    ret[nonzero_n] = nonzero_a
    return ret


class Categorical(Distribution):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        return F.sum(
            old_prob_var * (F.log(old_prob_var + TINY) - F.log(new_prob_var + TINY)),
            axis=-1,
        )

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info["prob"]
        new_prob = new_dist_info["prob"]
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=-1
        )

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        x_var = F.cast(x_var, np.float32)
        # Assume layout is N * A
        return (F.sum(new_prob_var * x_var, axis=-1) + TINY) / \
               (F.sum(old_prob_var * x_var, axis=-1) + TINY)

    def entropy_sym(self, dist_info_vars):
        probs = dist_info_vars["prob"]
        return -F.sum(probs * F.log(probs + TINY), axis=-1)

    def cross_entropy_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_prob_var = old_dist_info_vars["prob"]
        new_prob_var = new_dist_info_vars["prob"]
        # Assume layout is N * A
        return F.sum(
            old_prob_var * (- F.log(new_prob_var + TINY)),
            axis=-1
        )

    def entropy(self, info):
        probs = info["prob"]
        return -np.sum(probs * np.log(probs + TINY), axis=-1)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        return F.log(F.sum(probs * F.cast(x_var, np.float32), axis=-1) + TINY)

    def log_likelihood(self, xs, dist_info):
        probs = dist_info["prob"]
        # Assume layout is N * A
        return np.log(np.sum(probs * xs, axis=-1) + TINY)

    @property
    def dist_info_specs(self):
        return [("prob", (self.dim,))]

    def sample_sym(self, dist_info):
        probs = dist_info["prob"]
        samples = special.weighted_sample_n(probs.data, np.arange(self.dim))
        return chainer.Variable(samples)
