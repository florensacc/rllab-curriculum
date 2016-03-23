import theano.tensor as TT
import theano
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algo.batch_polopt import BatchPolopt
from rllab.optimizer.first_order_optimizer import FirstOrderOptimizer


class VPG(BatchPolopt):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            optimizer=None,
            **kwargs):
        if optimizer is None:
            optimizer_args = ext.merge_dict(
                dict(
                    update_method='adam',
                    batch_size=None,
                    max_epochs=1,
                ),
                kwargs
            )
            optimizer = FirstOrderOptimizer(**optimizer_args)
        self._optimizer = optimizer
        super(VPG, self).__init__(**kwargs)

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
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)
        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)

        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            max_kl = TT.max(kl * valid_var)
        else:
            surr_obj = - TT.mean(logli * advantage_var)
            mean_kl = TT.mean(kl)
            max_kl = TT.max(kl)

        input_list = [obs_var, action_var, advantage_var]
        if is_recurrent:
            input_list.append(valid_var)

        self._optimizer.update_opt(surr_obj, target=policy, inputs=input_list)

        f_kl = ext.compile_function(
            inputs=input_list + old_dist_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        return dict(
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        if policy.recurrent:
            inputs += (samples_data["valids"],)
        agent_infos = samples_data["agent_infos"]
        dist_info_list = [agent_infos[k] for k in policy.distribution.dist_info_keys]
        loss_before = self._optimizer.loss(inputs)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = opt_info['f_kl'](*(list(inputs) + dist_info_list))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, env, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            env=env,
        )
