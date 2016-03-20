from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algo.recurrent.recurrent_batch_polopt import RecurrentBatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer


class RPPO(RecurrentBatchPolopt):
    """
    Recurrent Proximal Policy Optimization.
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
        super(RPPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        obs_var = ext.new_tensor(
            'obs',
            ndim=2+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.matrix('advantage')
        old_pdist_var = ext.new_tensor(
            'old_pdist',
            ndim=3,
            dtype=theano.config.floatX
        )
        action_var = ext.new_tensor(
            'action',
            ndim=3,
            dtype=mdp.action_dtype
        )
        valid_var = TT.matrix('valid')
        pdist_var = policy.get_pdist_sym(obs_var, action_var)
        kl = policy.kl(old_pdist_var, pdist_var)
        lr = policy.likelihood_ratio(old_pdist_var, pdist_var, action_var)
        mean_kl = TT.sum(valid_var * kl) / TT.sum(valid_var)
        # formulate as a minimization problem
        surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        input_list = [
            obs_var,
            advantage_var,
            old_pdist_var,
            action_var,
            valid_var,
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
        all_input_values = list(ext.extract(
            samples_data,
            "observations", "advantages", "pdists", "actions", "valids"
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
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data, opt_info, **kwargs):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
