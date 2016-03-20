from rllab.misc.tensor_utils import flatten_tensors
from rllab.misc.ext import merge_dict, compile_function, extract, new_tensor
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab.algo.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from pydoc import locate
import numpy as np
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer


class PPO(BatchPolopt):
    """
    Proximal Policy Optimization.
    """

    def __init__(
            self,
            step_size=0.01,
            optimizer=None,
            **kwargs):
        if optimizer is None:
            optimizer = PenaltyLbfgsOptimizer()
        self._optimizer = optimizer
        self._step_size = step_size
        super(PPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        input_var = new_tensor(
            'input',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        old_pdist_var = TT.matrix('old_pdist')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)
        pdist_var = policy.get_pdist_sym(input_var)
        kl = policy.kl(old_pdist_var, pdist_var)
        lr = policy.likelihood_ratio(old_pdist_var, pdist_var, action_var)
        mean_kl = TT.mean(kl)
        # formulate as a minimization problem
        surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
            input_var,
            advantage_var,
            old_pdist_var,
            action_var,
        ]

        self._optimizer.update_opt(
            loss=surr_loss,
            target=policy,
            leq_constraint=(mean_kl, self._step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )

        return dict()

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        all_input_values = tuple(extract(
            samples_data,
            "observations", "advantages", "pdists", "actions"
        ))
        loss_before = self._optimizer.loss(all_input_values)
        self._optimizer.optimize(all_input_values)
        mean_kl = self._optimizer.constraint_val(all_input_values)
        loss_after = self._optimizer.loss(all_input_values)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data, opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
