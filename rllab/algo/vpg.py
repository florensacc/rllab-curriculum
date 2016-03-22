import theano.tensor as TT
import theano
from collections import OrderedDict
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod


class VPG(BatchPolopt, FirstOrderMethod):
    """
    Vanilla Policy Gradient.
    """

    def __init__(
            self,
            **kwargs):
        super(VPG, self).__init__(**kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

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

        logli = policy.log_likelihood_sym(obs_var, action_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        info_vars = policy.info_sym(obs_var, action_var)
        kl = policy.kl_sym(old_info_vars, info_vars)

        if is_recurrent:
            surr_obj = - TT.sum(logli * advantage_var * valid_var) / TT.sum(valid_var)
            mean_kl = TT.sum(kl * valid_var) / TT.sum(valid_var)
            max_kl = TT.max(kl * valid_var)
        else:
            surr_obj = - TT.mean(logli * advantage_var)
            mean_kl = TT.mean(kl)
            max_kl = TT.max(kl)

        updates = self.update_method(
            surr_obj, policy.get_params(trainable=True))
        assert isinstance(updates, OrderedDict)

        updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.iteritems()])
        input_list = [obs_var, action_var, advantage_var]

        f_update = ext.compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        f_loss = ext.compile_function(
            inputs=input_list,
            outputs=surr_obj,
        )
        f_kl = ext.compile_function(
            inputs=input_list + old_info_vars_list,
            outputs=[mean_kl, max_kl],
        )
        return dict(
            f_update=f_update,
            f_loss=f_loss,
            f_kl=f_kl,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        f_update = opt_info["f_update"]
        f_loss = opt_info["f_loss"]
        inputs = ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        )
        agent_infos = samples_data["agent_infos"]
        info_list = [agent_infos[k] for k in policy.info_keys]
        loss_before = f_loss(*inputs)
        f_update(*inputs)
        loss_after = f_loss(*inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = opt_info['f_kl'](*(list(inputs) + info_list))
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
