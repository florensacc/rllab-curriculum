from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algo.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
import theano.tensor as TT
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
    def init_opt(self, env_spec, policy, baseline):
        is_recurrent = int(policy.is_recurrent)
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
        old_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in policy.info_keys
            }
        old_info_vars_list = [old_info_vars[k] for k in policy.info_keys]


        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None

        info_vars = policy.info_sym(obs_var, action_var)
        kl = policy.kl_sym(old_info_vars, info_vars)
        lr = policy.likelihood_ratio_sym(action_var, old_info_vars, info_vars)
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
                     ] + old_info_vars_list
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
        info_list = [agent_infos[k] for k in policy.info_keys]
        all_input_values += tuple(info_list)
        if policy.is_recurrent:
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
