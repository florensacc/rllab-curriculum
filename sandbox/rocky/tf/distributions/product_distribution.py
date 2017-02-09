from sandbox.rocky.tf.distributions.base import Distribution
import numpy as np
import tensorflow as tf

from sandbox.rocky.tf.distributions.categorical import Categorical


class ProductDistribution(Distribution):
    def __init__(self, distributions):
        self.distributions = distributions
        self.dimensions = [x.dim for x in self.distributions]
        self._dim = sum(self.dimensions)

    @property
    def dim(self):
        return self._dim

    def _split_x(self, x):
        """
        Split the tensor variable or value into per component.
        """
        cum_dims = list(np.cumsum(self.dimensions))
        out = []
        for slice_from, slice_to, dist in zip([0] + cum_dims, cum_dims, self.distributions):
            sliced = x[:, slice_from:slice_to]
            if isinstance(dist, Categorical):
                if isinstance(sliced, (tf.Variable, tf.Tensor)):
                    sliced = tf.cast(sliced, tf.uint8)
                else:
                    sliced = np.cast['uint8'](sliced)
            out.append(sliced)
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

    def log_likelihood(self, xs, dist_infos):
        splitted_xs = self._split_x(xs)
        dist_infos = self._split_dist_info(dist_infos)
        ret = 0
        for x_i, dist_info_i, dist_i in zip(splitted_xs, dist_infos, self.distributions):
            ret += dist_i.log_likelihood(x_i, dist_info_i)
        return ret

    def log_likelihood_sym(self, x_var, dist_info_vars):
        splitted_x_vars = self._split_x(x_var)
        dist_info_vars = self._split_dist_info(dist_info_vars)
        ret = 0
        for x_var_i, dist_info_var_i, dist_i in zip(splitted_x_vars, dist_info_vars, self.distributions):
            ret += dist_i.log_likelihood_sym(x_var_i, dist_info_var_i)
        return ret

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        splitted_x_vars = self._split_x(x_var)
        old_dist_info_vars = self._split_dist_info(old_dist_info_vars)
        new_dist_info_vars = self._split_dist_info(new_dist_info_vars)
        ret = 1
        for x_var_i, old_dist_info_i, new_dist_info_i, dist_i in \
                zip(splitted_x_vars, old_dist_info_vars, new_dist_info_vars, self.distributions):
            ret *= dist_i.likelihood_ratio_sym(x_var_i, old_dist_info_i, new_dist_info_i)
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
    def dist_info_specs(self):
        ret = []
        for idx, dist in enumerate(self.distributions):
            for k, shape in dist.dist_info_specs:
                ret.append(("id_%d_%s" % (idx, k), shape))
        return ret

    def sample(self, dist_info):
        # get the components
        dist_infos = self._split_dist_info(dist_info)
        components = []
        for dist_i, dist_info_i in zip(self.distributions, dist_infos):
            components.append(dist_i.sample(dist_info_i))
        return list(zip(*components))

    def entropy(self, dist_info):
        # get the components
        dist_infos = self._split_dist_info(dist_info)
        ent = 0
        for dist_i, dist_info_i in zip(self.distributions, dist_infos):
            ent += dist_i.entropy(dist_info_i)
        return ent

    def entropy_sym(self, dist_info):
        # get the components
        dist_infos = self._split_dist_info(dist_info)
        ent = 0
        for dist_i, dist_info_i in zip(self.distributions, dist_infos):
            ent += dist_i.entropy_sym(dist_info_i)
        return ent

    def maximum_a_posteriori(self, dist_info):
        components = []
        for dist_i, dist_info_i in zip(self.distributions, self._split_dist_info(dist_info)):
            components.append(dist_i.maximum_a_posteriori(dist_info_i))
        return list(zip(*components))
