import numpy as np

from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.parallel.natural_gradient_method import \
    NaturalGradientMethod, master_f
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger


class TRPO(NaturalGradientMethod, BatchPolopt):
    """
    Trust Region Policy Optimization
    """

    @autoargs.inherit(NaturalGradientMethod.__init__)
    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.arg("backtrack_ratio", type=float,
                  help="The exponential backtrack factor")
    @autoargs.arg("max_backtracks", type=int,
                  help="The maximum number of exponential backtracks")
    def __init__(self,
                 backtrack_ratio=0.5,
                 max_backtracks=10,
                 **kwargs):
        BatchPolopt.__init__(self, algorithm_parallelized=True, **kwargs)
        NaturalGradientMethod.__init__(self, **kwargs)
        self.opt.backtrack_ratio = backtrack_ratio
        self.opt.max_backtracks = max_backtracks

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        with self.optimization_setup(itr, policy, samples_data, opt_info) as \
                flat_descent_step:
            logger.log("performing backtracking")
            prev_loss, prev_mean_kl, prev_max_kl = master_f("f_trpo_info")()
            prev_param = policy.get_param_values(trainable=True)
            for n_iter, ratio in enumerate(
                    self.opt.backtrack_ratio **
                    np.arange(self.opt.max_backtracks)):
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                policy.set_param_values(cur_param, trainable=True)
                parallel_sampler.master_set_param_values(
                    cur_param, trainable=True)
                loss, mean_kl, max_kl = master_f("f_trpo_info")()
                if loss < prev_loss and max_kl <= self.opt.step_size:
                    break
            logger.log("backtracking finished")
            logger.record_tabular('BacktrackItr', n_iter)
            logger.record_tabular('MeanKL', mean_kl)
            logger.record_tabular('MaxKL', max_kl)

        return opt_info
