import numpy as np
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides


class ParallelZeroBaseline(Baseline):

    def __init__(self, env_spec):
        pass

    @overrides
    def get_param_values(self, **kwargs):
        return None

    @overrides
    def set_param_values(self, val, **kwargs):
        pass

    @overrides
    def fit(self, paths):
        pass

    @overrides
    def predict(self, path):
        return np.zeros_like(path["rewards"])

    def init_par_objs(self,n_parallel):
        pass

    def init_rank(self,rank):
        self.rank = rank
