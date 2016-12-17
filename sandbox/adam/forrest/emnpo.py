from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer


class EMNPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        super(EMNPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = self.policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=1 + dim + is_recurrent,  # this is a change from vpg
                dtype=theano.config.floatX
            ) for k, dim in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k, _ in dist.dist_info_keys]

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=1 + dim + is_recurrent,  # this is a change from vpg
                dtype=theano.config.floatX
            ) for k, dim in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k, _ in self.policy.state_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        lr = dist.sampled_likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        entropy_new = dist.importance_entropy_sym(action_var, old_dist_info_vars, dist_info_vars)
        # entropy_new = dist.conditional_entropy_sym(action_var, dist_info_vars)
        entropy_old = 1.0  # dist.conditional_entropy_sym(action_var, old_dist_info_vars)

        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
            performance_constraint = -TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
            entropy_term = TT.mean(entropy_new) / TT.mean(entropy_old)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(logli * advantage_var)
            performance_constraint = -TT.mean(lr * advantage_var)
            entropy_term = TT.mean(entropy_new)  # / TT.mean(entropy_old)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + state_info_vars_list + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            kl_constraint=(mean_kl, self.step_size),
            entropy_term=entropy_term,
            performance_constraint=(performance_constraint, 0.0),
            # leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k, _ in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k, _ in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self.optimizer.loss(all_input_values)
        mean_kl_before, conditional_entropy_before, performance_before = self.optimizer.constraint_val(all_input_values)
        # mean_kl_before = self.optimizer.constraint_val(all_input_values)
        self.optimizer.optimize(all_input_values)
        mean_kl, conditional_entropy, performance = self.optimizer.constraint_val(all_input_values)
        # mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('PerformanceBefore', performance_before)
        logger.record_tabular('Performance', performance)
        logger.record_tabular('ConditionalEntropyBefore', conditional_entropy_before)
        logger.record_tabular('ConditionalEntropy', conditional_entropy)
        # logger.record_tabular('PerformanceBefore', mean_kl_before)
        # logger.record_tabular('Performance', mean_kl)
        # logger.record_tabular('ConditionalEntropyBefore', mean_kl_before)
        # logger.record_tabular('ConditionalEntropy', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
