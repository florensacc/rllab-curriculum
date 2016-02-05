import theano.tensor as TT
from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, new_tensor
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod


class VPG(BatchPolopt, FirstOrderMethod):
    """
    Vanilla Policy Gradient.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    def __init__(
            self,
            **kwargs):
        super(VPG, self).__init__(**kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

    @overrides
    def init_opt(self, mdp, policy, baseline):
        obs_var = new_tensor(
            'obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)
        log_prob = policy.get_log_prob_sym(obs_var, action_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(log_prob * advantage_var)
        updates = self.update_method(
            surr_obj, policy.get_params(trainable=True))
        input_list = [obs_var, advantage_var, action_var]

        old_pdist_var = TT.matrix('old_pdist')
        pdist_var = policy.get_pdist_sym(obs_var)
        kl = policy.kl(old_pdist_var, pdist_var)
        mean_kl = TT.mean(kl)
        max_kl = TT.max(kl)

        f_update = compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        f_loss = compile_function(
            inputs=input_list,
            outputs=surr_obj,
        )
        f_kl = compile_function(
            inputs=input_list + [old_pdist_var],
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
        inputs = extract(
            samples_data,
            "observations", "advantages", "actions"
        )
        pdists = samples_data["pdists"]
        loss_before = f_loss(*inputs)
        f_update(*inputs)
        loss_after = f_loss(*inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)

        mean_kl, max_kl = opt_info['f_kl'](*(list(inputs) + [pdists]))
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('MaxKL', max_kl)

        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data,
                         opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )

