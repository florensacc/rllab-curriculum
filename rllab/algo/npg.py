import tensorfuse as theano
import tensorfuse.tensor as TT
from rllab.misc import logger, autoargs
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod
import cPickle as pickle


class NPG(BatchPolopt, FirstOrderMethod):
    """
    Natural Policy Gradient.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    def __init__(
            self,
            **kwargs):
        super(NPG, self).__init__(**kwargs)
        FirstOrderMethod.__init__(self, **kwargs)

    @overrides
    def init_opt(self, mdp, policy, vf):
        input_var = policy.input_var
        advantage_var = TT.vector('advantage')
        action_var = policy.new_action_var('action')
        ref_policy = pickle.loads(pickle.dumps(policy))
        ref_input_var = ref_policy.input_var

        log_prob = policy.get_log_prob_sym(action_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.mean(log_prob * advantage_var)
        # We would need to calculate the empirical fisher information matrix
        # This can be done as follows (though probably not the most efficient
        # way):
        # Since I(theta) = d^2 KL(p(theta)||p(theta')) / d theta^2
        #                  (evaluated at theta' = theta),
        # we can get I(theta) by calculating the hessian of
        # KL(p(theta)||p(theta'))
        mean_kl = TT.mean(policy.kl(policy.pdist_var, ref_policy.pdist_var))
        # Here, we need to ensure that all the parameters are flattened
        emp_fishers = theano.gradient.hessian(mean_kl, wrt=policy.params)
        grads = theano.grad(surr_obj, wrt=policy.params)
        # Is there a better name...
        fisher_grads = []
        for emp_fisher, grad in zip(emp_fishers, grads):
            reg_fisher = emp_fisher + TT.eye(emp_fisher.shape[0])
            inv_fisher = TT.nlinalg.matrix_inverse(reg_fisher)
            fisher_grads.append(inv_fisher.dot(grad))

        updates = self.update_method(fisher_grads, policy.params)

        input_list = [input_var, advantage_var, action_var, ref_input_var]
        f_update = compile_function(
            inputs=input_list,
            outputs=None,
            updates=updates,
        )
        f_loss = compile_function(
            inputs=input_list,
            outputs=surr_obj,
        )
        return dict(
            f_update=f_update,
            f_loss=f_loss,
            ref_policy=ref_policy,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        cur_params = policy.get_param_values()
        logger.log("optimizing policy")
        f_update = opt_info["f_update"]
        f_loss = opt_info["f_loss"]
        ref_policy = opt_info["ref_policy"]
        inputs = extract(
            samples_data,
            "observations", "advantages", "actions", "observations"
        )
        # Need to ensure this
        ref_policy.set_param_values(cur_params)
        logger.log("computing loss before")
        loss_before = f_loss(*inputs)
        logger.log("performing update")
        f_update(*inputs)
        logger.log("computing loss after")
        loss_after = f_loss(*inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.log("optimization finished")
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, vf, samples_data, opt_info):
        return dict(
            itr=itr,
            policy=policy,
            vf=vf,
            mdp=mdp,
            observations=samples_data["observations"],
            advantages=samples_data["advantages"],
            actions=samples_data["actions"],
        )
