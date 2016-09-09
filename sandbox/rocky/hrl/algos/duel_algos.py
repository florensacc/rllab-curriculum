


import numpy as np
from rllab.misc import special
from rllab.misc import ext
from rllab.misc import logger
from rllab.algos.batch_polopt import BatchPolopt
from rllab.misc import tensor_utils
from rllab.misc.overrides import overrides
from rllab.algos.npo import NPO
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable
from rllab.sampler.stateful_pool import singleton_pool
from rllab.sampler.utils import rollout
import theano
import theano.tensor as TT
import pickle as pickle

floatX = theano.config.floatX


class DuelBatchPolopt(BatchPolopt, Serializable):
    def __init__(
            self,
            master_algo,
            duel_algo,
            *args,
            **kwargs):
        """
        :type master_algo: BatchPolopt
        :type duel_algo: BatchPolopt
        """
        Serializable.quick_init(self, locals())
        self.master_algo = master_algo
        self.duel_algo = duel_algo
        super(DuelBatchPolopt, self).__init__(*args, **kwargs)

    def train(self):
        if self.plot:
            self.master_algo.plot = True
        with logger.tabular_prefix('Master_'), logger.prefix('Master | '):
            self.master_algo.start_worker()
            self.master_algo.init_opt()
        with logger.tabular_prefix('Duel_'), logger.prefix('Duel | '):
            self.duel_algo.start_worker()
            self.duel_algo.init_opt()

        for itr in range(self.start_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                with logger.tabular_prefix('Master_'), logger.prefix('Master | '):
                    master_paths = self.master_algo.obtain_samples(itr)
                    master_samples_data = self.master_algo.process_samples(itr, master_paths)
                    self.master_algo.log_diagnostics(master_paths)
                    self.master_algo.optimize_policy(itr, master_samples_data)
                with logger.tabular_prefix('Duel_'), logger.prefix('Duel | '):
                    duel_paths = self.duel_algo.obtain_samples(itr)
                    duel_samples_data = self.duel_algo.process_samples(itr, duel_paths)
                    self.duel_algo.log_diagnostics(duel_paths)
                    self.duel_algo.optimize_policy(itr, duel_samples_data)
                logger.log("saving snapshot...")
                master_params = self.master_algo.get_itr_snapshot(itr, master_samples_data)  # , **kwargs)
                duel_params = self.duel_algo.get_itr_snapshot(itr, duel_samples_data)  # , **kwargs)
                if self.store_paths:
                    master_params["paths"] = master_samples_data["paths"]
                    duel_params["paths"] = duel_samples_data["paths"]
                logger.save_itr_params(itr, dict(master=master_params, duel=duel_params, **master_params))
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.master_algo.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

        self.master_algo.shutdown_worker()
        self.duel_algo.shutdown_worker()
