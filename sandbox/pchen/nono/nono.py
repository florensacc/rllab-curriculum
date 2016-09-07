import numpy as np

from rllab.algos.npo import NPO
from rllab.misc import autoargs
from rllab.misc.ext import extract
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger

class NONO(NPO):
    """
    Near Off-policy Near Online TRPO
    """

    def __init__(self,
                 backtrack_ratio=0.5,
                 max_backtracks=10,
                 trust_region=0.1,
                 **kwargs):
        super(NONO, self).__init__(**kwargs)
        self.trust_region = trust_region
        self.backtrack_ratio = backtrack_ratio
        self.max_backtracks = max_backtracks
        self.memory = []

    @overrides
    def optimize_policy(self, itr, samples_data):
        policy = self.policy
        self.memory.insert(0, samples_data)
        f_trpo_info = opt_info['f_trpo_info']
        to_delete = []
        used = 0
        for idx, data in enumerate(self.memory):
            inputs = list(extract(
                data,
                "observations", "advantages", "pdists", "actions"
            ))
            prev_loss, prev_mean_kl, prev_max_kl = f_trpo_info(*inputs)

            if prev_mean_kl >= 5 * self.trust_region:
                to_delete.append(idx)

            if prev_mean_kl <= self.trust_region:
                print("%s using... %s" % (idx, prev_mean_kl))
                with self.optimization_setup(itr, policy, data, opt_info) as (
                        inputs, flat_descent_step):
                    logger.log("performing backtracking")
                    prev_param = policy.get_param_values(trainable=True)
                    succeed = False
                    for n_iter, ratio in enumerate(self.backtrack_ratio ** np.arange(self.max_backtracks)):
                        cur_step = ratio * flat_descent_step
                        cur_param = prev_param - cur_step
                        policy.set_param_values(cur_param, trainable=True)
                        loss, mean_kl, max_kl = f_trpo_info(*inputs)
                        if loss < prev_loss and (mean_kl-prev_mean_kl) <= self.step_size and mean_kl <= self.trust_region:
                            succeed = True
                            break
                    if not succeed:
                        policy.set_param_values(prev_param, trainable=True)
                        logger.log("backtracking failed")
                    else:
                        logger.log("backtracking succeeded")
                        used += 1
                    logger.record_tabular('BacktrackItr', n_iter)
                    logger.record_tabular('BacktrackSuccess', succeed)
                    logger.record_tabular('MeanKL', mean_kl)
                    logger.record_tabular('MaxKL', max_kl)
            else:
                print("%s skipping... %s" % (idx, prev_mean_kl))
        logger.record_tabular('BatchesUsed', used)
        for id in reversed(to_delete):
            self.memory.pop(id)

        return opt_info
