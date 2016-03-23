from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algo.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer


class NPO(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer,
            step_size=0.01,
            **kwargs):
        if optimizer is None:
            optimizer = PenaltyLbfgsOptimizer()
        self._optimizer = optimizer
        self._step_size = step_size
        super(NPO, self).__init__(**kwargs)

    @overrides
    def init_opt(self, env_spec, policy, baseline):
        is_recurrent = int(policy.recurrent)
        obs_var = env_spec.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = env_spec.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        dist = policy.distribution
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        dist_info_vars = policy.dist_info_sym(obs_var, action_var)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
        if is_recurrent:
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            surr_loss = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(kl)
            surr_loss = - TT.mean(lr * advantage_var)

        input_list = [
                         obs_var,
                         action_var,
                         advantage_var,
                     ] + old_dist_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

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
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        info_list = [agent_infos[k] for k in policy.distribution.dist_info_keys]
        all_input_values += tuple(info_list)
        if policy.recurrent:
            all_input_values += (samples_data["valids"],)
        loss_before = self._optimizer.loss(all_input_values)
        self._optimizer.optimize(all_input_values)
        mean_kl = self._optimizer.constraint_val(all_input_values)
        loss_after = self._optimizer.loss(all_input_values)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, env, policy, baseline, samples_data, opt_info, **kwargs):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            env=env,
        )