import theano.tensor as TT
import numpy as np
from rllab.distributions.base import Distribution
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

TINY = 1e-8


# def from_onehot_sym(x_var):
#     ret = TT.zeros((x_var.shape[0],), x_var.dtype)
#     nonzero_n, nonzero_a = TT.nonzero(x_var)[:2]
#     ret = TT.set_subtensor(ret[nonzero_n], nonzero_a.astype('uint8'))
#     return ret


# def from_onehot(x_var): # this is only for one onehot vector (no array of one-hots)!
#     ret = np.zeros((len(x_var),), 'int32')
#     nonzero_n, nonzero_a = np.nonzero(x_var)
#     ret[nonzero_n] = nonzero_a
#     return ret

# see misc.special.py for "from onehot/sym" function

# return an array with ONE LESS axis than x_var, as the last one has been collapsed to the corresp index
def from_onehot(x_var):
    ret = np.zeros(np.shape(x_var)[:-1], 'int32')
    nonzero_indexes = np.nonzero(x_var)
    ret[nonzero_indexes[:-1]] = nonzero_indexes[-1]
    return ret


def from_index(x_var, dim):  # I need to know the dim of the one-hot to output. [1,0,0..] corresp to 0!!!! (not 1)
    if type(x_var) is np.ndarray and x_var.shape and x_var.shape[0] > 1:
        new_array = []
        for row in x_var:
            new_array.append(from_index(row, dim))
        return np.stack(new_array)

    else:
        ret = np.zeros(dim, 'int32')
        ret[x_var] = 1
        return ret


class Categorical(Distribution):  # modified version where the array "prob" has only one axis
    def __init__(self, dim):
        self._dim = dim
        self._srng = RandomStreams()

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
        return TT.sum(
            old_prob_var * (TT.log(old_prob_var + TINY) - TT.log(new_prob_var + TINY)),
            axis=-1
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
        x_var = TT.cast(x_var, 'float32')
        # Assume layout is N * A
        return (TT.sum(new_prob_var * x_var, axis=-1) + TINY) / (TT.sum(old_prob_var * x_var, axis=-1) + TINY)

    def entropy(self, info):
        probs = info["prob"]
        return -np.sum(probs * np.log(probs + TINY), axis=-1)  # I also changed to -1 here

    def entropy_sym(self, dist_info_vars):
        prob_var = dist_info_vars["prob"]
        return -TT.sum(prob_var * TT.log(prob_var + TINY), axis=-1)  # changed to -1

    def log_likelihood_sym(self, x_var, dist_info_vars):
        probs = dist_info_vars["prob"]
        # Assume layout is N * A  ## not really! the output will just be (n1,n2,..) And when using this for fitting,
        return TT.log(TT.sum(probs * TT.cast(x_var, 'float32'), axis=-1) + TINY)  # there is a TT.mean, and because it
        # is just a max, the 1/n constant does not affect

    def log_likelihood(self, xs, dist_info):  # layout is (n1,n2,...,A)
        probs = np.asarray(dist_info["prob"])
        # this does not affect 2D arrays, but converts the 3D in flatten but the last dim
        probs_quasi_flat = probs.reshape(-1, probs.shape[-1])
        xs_quasi_flat = np.asarray(xs).reshape(-1, xs.shape[-1])
        N = probs_quasi_flat.shape[0]
        # print 'probs_quasi_flat: ', probs_quasi_flat, '\nxs_quasi_flat: ', xs_quasi_flat, \
        #     '\nlikelihood: ', probs_quasi_flat[np.arange(N), from_onehot(xs_quasi_flat)]
        # Assume layout is N * A  # changed!
        return np.log(probs_quasi_flat[np.arange(N), from_onehot(xs_quasi_flat)] + TINY)

    def sample_sym(self, dist_info):
        probs = dist_info["prob"]
        return self._srng.multinomial(pvals=probs, dtype='uint8')

    def sample(self, dist_info):
        probs = dist_info["prob"]  # here this has to be of shape (dim,) OR (dim, 1); NO whatever larger than 1!!
        if isinstance(probs[0], (list, tuple, np.ndarray)):
            probs = probs[0]
        # samples = []  # if we need to input sevaral distributions at the same time (along axis 0 of probs)
        # for prob in probs:
        #     samples.append(np.random.multinomial(n=1, pvals=prob, size=1))
        # return samples
        return np.random.multinomial(n=1, pvals=probs)  # , size=1)  # this gives a one-hot of shape (1, dim)

    @property
    def dist_info_keys(self):
        return ["prob"]
