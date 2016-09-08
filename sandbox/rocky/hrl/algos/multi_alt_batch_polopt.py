


import numpy as np
from rllab.misc import special
from rllab.misc import ext
from rllab.misc import logger
from rllab.algos.base import RLAlgorithm
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


class MultiAltBatchPolopt(RLAlgorithm, Serializable):
    def __init__(
            self,
            algos,
            n_itr=500,
            *args,
            **kwargs):
        Serializable.quick_init(self, locals())
        self.algos = [(x["algo"], x["name"]) for _, x in sorted(algos.items())]
        self.n_itr = n_itr

    def train(self):
        for algo, name in self.algos:
            with logger.tabular_prefix('%s_' % name), logger.prefix('%s | ' % name):
                algo.start_worker()
                algo.init_opt()

        for itr in range(self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                all_samples_data = dict()
                for algo, name in self.algos:
                    with logger.tabular_prefix('%s_' % name), logger.prefix('%s | ' % name):
                        paths = algo.obtain_samples(itr)
                        samples_data = algo.process_samples(itr, paths)
                        algo.log_diagnostics(paths)
                        algo.optimize_policy(itr, samples_data)
                        all_samples_data[name] = samples_data
                logger.log("saving snapshot...")
                all_params = None
                for algo, name in self.algos:
                    algo_params = algo.get_itr_snapshot(itr, all_samples_data[name])
                    if all_params is None:
                        all_params = algo_params
                    all_params[name] = algo_params
                logger.save_itr_params(itr, all_params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)

        for algo, _ in self.algos:
            algo.shutdown_worker()
