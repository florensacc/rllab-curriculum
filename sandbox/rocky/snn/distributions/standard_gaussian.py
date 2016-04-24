from __future__ import print_function
from __future__ import absolute_import
import theano.tensor as TT
import numpy as np
from rllab.distributions.base import Distribution


class StandardGaussian(Distribution):
    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        return TT.zeros_like(TT.sum(old_dist_info_vars["shape_placeholder"], axis=-1))

    def kl(self, old_dist_info, new_dist_info):
        return np.zeros_like(np.sum(old_dist_info["shape_placeholder"], axis=-1))

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        return TT.ones_like(TT.sum(old_dist_info_vars["shape_placeholder"], axis=-1))

    def log_likelihood_sym(self, x_var, dist_info_vars):
        return - 0.5 * TT.sum(TT.square(x_var), axis=-1) - \
               0.5 * x_var.shape[-1] * np.log(2 * np.pi)

    def log_likelihood(self, xs, dist_info):
        return - 0.5 * np.sum(np.square(xs), axis=-1) - \
               0.5 * xs.shape[-1] * np.log(2 * np.pi)

    def entropy(self, dist_info):
        return np.sum(np.zeros_like(dist_info["shape_placeholder"]) + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_info_keys(self):
        return ["shape_placeholder"]
