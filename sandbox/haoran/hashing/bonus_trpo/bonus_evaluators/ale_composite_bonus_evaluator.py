import numpy as np
import multiprocessing as mp
from rllab.misc import logger
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.ale_hashing_bonus_evaluator import ALEHashingBonusEvaluator
from sandbox.adam.parallel.util import SimpleContainer


class ALECompositeBonusEvaluator(object):
    """
    Use a collection of bonus evaluators to compute bonuses on the same type of count targets. Then sum up the bonuses.
    The current implementation can be improved by:
    1. Preprocessing the states before feeding them into subordinate evaluators
    2. Reduce barriers in subordinate evaluators, by allowing the same evaluator to not wait for other parallel copies, but instead proceeed to other evaluators.
    """
    def __init__(
            self,
            bonus_evaluators,
            log_prefix="",
            parallel=False,
        ):
        self.bonus_evaluators = bonus_evaluators
        self.log_prefix = log_prefix
        self.parallel = parallel
        for ev in bonus_evaluators:
            assert parallel == ev.parallel

        # logging stats ---------------------------------
        self.rank = None


    def init_rank(self,rank):
        self.rank = rank
        for ev in self.bonus_evaluators:
            ev.init_rank(rank)

    def init_par_objs(self,n_parallel):
        for ev in self.bonus_evaluators:
            ev.init_par_objs(n_parallel)


    def fit_before_process_samples(self, paths):
        for ev in self.bonus_evaluators:
            ev.fit_before_process_samples(paths)

    def predict(self, path):
        bonuses = np.zeros(len(path["rewards"]))
        for ev in self.bonus_evaluators:
            bonuses = bonuses + ev.predict(path)

        return bonuses

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        pass
