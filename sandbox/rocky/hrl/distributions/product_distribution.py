from __future__ import print_function
from __future__ import absolute_import
from rllab.distributions.base import Distribution
import numpy as np


class ProductDistribution(Distribution):
    def __init__(self, distributions, dimensions):
        self.distributions = distributions
        self.dimensions = dimensions

    def _split_x(self, x):
        """
        Split the tensor variable or value into per component.
        """
        cum_dims = list(np.cumsum(self.dimensions))
        out = []
        for slice_from, slice_to in zip([0] + cum_dims, cum_dims):
            out.append(x[:, slice_from:slice_to])
        return out

    def _split_dist_info(self, dist_info):
        """
        Split the dist info dictionary into per component.
        """
        ret = []
        for idx, dist in enumerate(self.distributions):
            cur_dist_info = dict()
            for k in dist.dist_info_keys:
                cur_dist_info[k] = dist_info["id_%d_%s" % (idx, k)]
            ret.append(cur_dist_info)
        return ret

    def log_likelihood_sym(self, x_var, dist_info_vars):
        splitted_x_vars = self._split_x(x_var)
        dist_info_vars = self._split_dist_info(dist_info_vars)
        ret = 0
        for x_var_i, dist_info_var_i, dist_i in zip(splitted_x_vars, dist_info_vars, self.distributions):
            ret += dist_i.log_likelihood_sym(x_var_i, dist_info_var_i)
        return ret

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        old_dist_info_vars = self._split_dist_info(old_dist_info_vars)
        new_dist_info_vars = self._split_dist_info(new_dist_info_vars)
        ret = 0
        for old_dist_info_var_i, new_dist_info_var_i, dist_i in zip(
                old_dist_info_vars, new_dist_info_vars, self.distributions):
            ret += dist_i.kl_sym(old_dist_info_var_i, new_dist_info_var_i)
        return ret

    def kl(self, old_dist_infos, new_dist_infos):
        old_dist_infos = self._split_dist_info(old_dist_infos)
        new_dist_infos = self._split_dist_info(new_dist_infos)
        ret = 0
        for old_dist_info_i, new_dist_info_i, dist_i in zip(
                old_dist_infos, new_dist_infos, self.distributions):
            ret += dist_i.kl(old_dist_info_i, new_dist_info_i)
        return ret

    @property
    def dist_info_keys(self):
        ret = []
        for idx, dist in enumerate(self.distributions):
            for k in dist.dist_info_keys:
                ret.append("id_%d_%s" % (idx, k))
        return ret
