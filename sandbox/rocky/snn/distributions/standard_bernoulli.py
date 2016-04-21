from __future__ import print_function
from __future__ import absolute_import
import theano.tensor as TT
import numpy as np
from rllab.distributions.base import Distribution


class StandardBernoulli(Distribution):
    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        return TT.zeros_like(TT.sum(old_dist_info_vars["shape_placeholder"], axis=-1))

    def kl(self, old_dist_info, new_dist_info):
        return np.zeros_like(np.sum(old_dist_info["shape_placeholder"], axis=-1))

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        return TT.ones_like(TT.sum(old_dist_info_vars["shape_placeholder"], axis=-1))

    def log_likelihood_sym(self, x_var, dist_info_vars):
        return np.log(0.5) * TT.sum(TT.ones_like(dist_info_vars["shape_placeholder"]), axis=-1)

    def log_likelihood(self, xs, dist_info):
        return np.log(0.5) * np.sum(np.ones_like(xs), axis=-1)

    def entropy(self, dist_info):
        return np.sum(np.log(2) * np.ones_like(dist_info["shape_placeholder"]), axis=-1)

    @property
    def dist_info_keys(self):
        return ["shape_placeholder"]
