import theano.tensor as TT
import numpy as np
from rllab.distributions.categorical import Categorical

TINY = 1e-8


# see misc.special.py for "from onehot/sym" function

# return an array with ONE LESS axis than x_var, as the last one has been collapsed to the corresp index
def from_onehot(x_var):
    ret = np.zeros(np.shape(x_var)[:-1], 'int32')
    nonzero_indexes = np.nonzero(x_var)
    ret[nonzero_indexes[:-1]] = nonzero_indexes[-1]
    return ret


def from_index(x_var, dim):  # I need to know the dim of the one-hot to output. [1,0,0..] corresp to 0 (not 1)
    if type(x_var) is np.ndarray and x_var.shape and x_var.shape[0] > 1:
        new_array = []
        for row in x_var:
            new_array.append(from_index(row, dim))
        return np.stack(new_array)

    else:
        ret = np.zeros(dim, 'int32')
        ret[x_var] = 1
        return ret


class Categorical_oneAxis(Categorical):
    """
    Modified version of rllab.distributions.categorical where the array "prob" has only one axis
    See original class for dim, kl_sym, kl, likelihood_ratio_sym, sample_sym, dist_info_keys
    """
    def __init__(self, dim):
        super(Categorical_oneAxis, self).__init__(dim)

    def entropy(self, info):
        probs = info["prob"]
        return -np.sum(probs * np.log(probs + TINY), axis=-1)  # I also changed to -1 here

    def entropy_sym(self, dist_info_vars):
        prob_var = dist_info_vars["prob"]
        return -TT.sum(prob_var * TT.log(prob_var + TINY), axis=-1)  # changed to -1

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        # Assume layout is N * A
        # not really! the output will just be (n1,n2,..) And when using this for fitting,
        # there is a TT.mean, and because it is just a max, the 1/n constant does not affect
        return TT.log(TT.sum(probs * TT.cast(x_var, 'float32'), axis=-1) + TINY)

    def log_likelihood(self, xs, dist_info):  # layout is (n1,n2,...,A)
        probs = np.asarray(dist_info["prob"])
        # this does not affect 2D arrays, but converts the 3D in flatten but the last dim
        probs_quasi_flat = probs.reshape(-1, probs.shape[-1])
        xs_quasi_flat = np.asarray(xs).reshape(-1, xs.shape[-1])
        N = probs_quasi_flat.shape[0]
        # Assume layout is N * A  # changed!
        return np.log(probs_quasi_flat[np.arange(N), from_onehot(xs_quasi_flat)] + TINY)

    def sample(self, dist_info):
        probs = dist_info["prob"]  # here this has to be of shape (dim,) OR (dim, 1); NO larger than 1!!
        if isinstance(probs[0], (list, tuple, np.ndarray)):
            probs = probs[0]
        return np.random.multinomial(n=1, pvals=probs)  # this gives a one-hot of shape (1, dim)

