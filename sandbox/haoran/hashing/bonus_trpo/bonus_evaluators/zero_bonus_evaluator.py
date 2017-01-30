import numpy as np
import itertools
from rllab.misc import logger


class ZeroBonusEvaluator(object):
    def __init__(self):
        pass

    def init_par_objs(self,n_parallel):
        pass

    def init_shared_dict(self, shared_dict):
        pass

    def init_rank(self,rank):
        self.rank = rank

    def fit_before_process_samples(self, paths):
        pass

    def fit_after_process_samples(self, samples_data):
        pass

    def predict(self, path):
        T = len(path["rewards"])
        bonuses = np.zeros(T)
        return bonuses

    def log_diagnostics(self, paths):
        pass
