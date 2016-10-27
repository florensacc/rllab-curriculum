import numpy as np
import multiprocessing as mp
from rllab.misc import logger
from sandbox.adam.parallel.util import SimpleContainer


class RandomBonusEvaluator(object):
    """
    Assigns a randomly drawn (positive) exploration reward in [0,1]
    """
    def __init__(
            self,
            log_prefix="",
            randomness_form="uniform",
        ):
        self.log_prefix = log_prefix
        self.randomness_form = randomness_form

    def init_rank(self,rank):
        pass

    def init_par_objs(self,n_parallel):
        pass

    def fit_before_process_samples(self, paths):
        pass

    def predict(self, path):
        n_steps = len(path["rewards"])
        if self.randomness_form == "uniform":
            bonuses = np.random.rand(n_steps)
        else:
            raise NotImplementedError

        return bonuses

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        pass
