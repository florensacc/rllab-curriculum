


from .base import DistributionExt


class Deterministic(DistributionExt):
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def dist_flat_dim(self):
        return self.dim

    def sample(self, dist_info):
        return dist_info["value"]

    def sample_sym(self, dist_info_sym):
        return dist_info_sym["value"]

    def activate_dist(self, dist_flat):
        return dict(value=dist_flat)

    def activate_dist_sym(self, dist_flat_sym):
        return dict(value=dist_flat_sym)

    @property
    def dist_info_keys(self):
        return ["value"]
