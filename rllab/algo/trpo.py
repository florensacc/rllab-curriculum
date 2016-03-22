import numpy as np

from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.natural_gradient_method import NaturalGradientMethod
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger


class TRPO(NaturalGradientMethod, BatchPolopt):
    """
    Trust Region Policy Optimization
    """

    def __init__(self,
                 backtrack_ratio=0.5,
                 max_backtracks=10,
                 **kwargs):
        """
        :param backtrack_ratio: The exponential backtrack factor.
        :param max_backtracks: The maximum number of exponential backtracks.
        :param kwargs:
        :return:
        """
        super(TRPO, self).__init__(**kwargs)
        BatchPolopt.__init__(self, **kwargs)
        self.backtrack_ratio = backtrack_ratio
        self.max_backtracks = max_backtracks

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        with self.optimization_setup(itr, policy, samples_data, opt_info) as (
                inputs, flat_descent_step):
            logger.log("performing backtracking")
            f_trpo_info = opt_info['f_trpo_info']
            prev_loss, prev_mean_kl, prev_max_kl = f_trpo_info(*inputs)
            prev_param = policy.get_param_values(trainable=True)
            for n_iter, ratio in enumerate(self.backtrack_ratio ** np.arange(self.max_backtracks)):
                cur_step = ratio * flat_descent_step
                cur_param = prev_param - cur_step
                policy.set_param_values(cur_param, trainable=True)
                loss, mean_kl, max_kl = f_trpo_info(*inputs)
                if loss < prev_loss and mean_kl <= self.step_size:
                    break
            logger.log("backtracking finished")
            logger.record_tabular('BacktrackItr', n_iter)
            logger.record_tabular('MeanKL', mean_kl)
            logger.record_tabular('MaxKL', max_kl)

        return opt_info

