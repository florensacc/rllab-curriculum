


from rllab.distributions.base import Distribution


class DistributionExt(Distribution):

    @property
    def dist_flat_dim(self):
        # number of units needed
        raise NotImplementedError

    def sample(self, dist_info):
        raise NotImplementedError

    def sample_sym(self, dist_info_sym):
        raise NotImplementedError

    def activate_dist(self, dist_flat):
        raise NotImplementedError

    def activate_dist_sym(self, dist_flat_sym):
        raise NotImplementedError
