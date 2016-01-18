from itertools import izip

import theano
import theano.tensor as TT
import numpy as np
from rllab.misc import logger, autoargs
from rllab.misc.console import Message
from rllab.misc.krylov import cg
from rllab.misc.overrides import overrides
from rllab.misc.ext import extract, compile_function, flatten_hessian, new_tensor, new_tensor_like, \
    flatten_tensor_variables
from rllab.algo.batch_polopt import BatchPolopt
from rllab.algo.first_order_method import FirstOrderMethod
import cPickle as pickle


class NPG(BatchPolopt, FirstOrderMethod):
    """
    Natural Policy Gradient.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    @autoargs.inherit(FirstOrderMethod.__init__)
    @autoargs.arg("step_size", type=float,
                  help="KL divergence constraint. Default to None, in which case"
                       "only natural gradient direction is calculated")
    def __init__(
            self,
            step_size=None,
            **kwargs):
        super(NPG, self).__init__(**kwargs)
        FirstOrderMethod.__init__(self, **kwargs)
        self.step_size = step_size

    @overrides
    def init_opt(self, mdp, policy, vf):
        input_var = new_tensor(
            'input',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.vector('advantage')
        action_var = TT.matrix('action', dtype=mdp.action_dtype)

        log_prob = policy.get_log_prob_sym(input_var, action_var)
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
        old_pdist_var = TT.matrix('old_pdist')
        pdist_var = policy.get_pdist_sym(input_var)
        mean_kl = TT.mean(policy.kl(old_pdist_var, pdist_var))
        grads = theano.grad(surr_obj, wrt=policy.params)
        flat_gard = flatten_tensor_variables(grads)

        kl_grads = theano.grad(mean_kl, wrt=policy.params)
        kl_flat_grad = flatten_tensor_variables(kl_grads)
        emp_fisher = flatten_hessian(mean_kl, wrt=policy.params, block_diagonal=False)
        xs = [
            new_tensor_like("%s x" % p.name, p)
            for p in policy.params
        ]
        Hx_rop = TT.sum(TT.Rop(kl_flat_grad, policy.params, xs), axis=0)
        Hx_plain_splits = TT.grad(TT.sum([
            g * x for g, x in izip(kl_grads, xs)
        ]), wrt=policy.params)
        Hx_plain = TT.concatenate([s.flatten() for s in Hx_plain_splits])

        input_list = [input_var, advantage_var, old_pdist_var, action_var]
        f_loss = compile_function(
            inputs=input_list,
            outputs=surr_obj,
        )
        f_grad = compile_function(
            inputs=input_list,
            outputs=[surr_obj, flat_gard],
        )
        f_fisher = compile_function(
            inputs=input_list,
            outputs=[surr_obj, flat_gard, emp_fisher],
        )
        f_Hx_rop = compile_function(
            inputs=input_list + xs,
            outputs=Hx_rop,
        )
        f_Hx_plain = compile_function(
            inputs=input_list + xs,
            outputs=Hx_plain,
        )

        descent_steps = [
            new_tensor_like("%s descent" % p.name, p)
            for p in policy.params
        ]
        updates = self.update_method(descent_steps, policy.params)
        f_update = compile_function(
            inputs=descent_steps,
            outputs=None,
            updates=updates,
        )

        # lazy dict?
        return dict(
            f_loss=f_loss,
            f_grad=f_grad,
            f_fisher=f_fisher,
            f_Hx_rop=f_Hx_rop,
            f_Hx_plain=f_Hx_plain,
            f_update=f_update,
        )

    @overrides
    def optimize_policy(self, itr, policy, samples_data, opt_info):
        cur_params = policy.get_param_values()
        logger.log("optimizing policy")
        f_loss, f_grad, f_fisher, f_Hx_rop, f_Hx_plain, f_update = \
            extract(opt_info, "f_loss", "f_grad", "f_fisher", "f_Hx_rop",
                    "f_Hx_plain", "f_update")
        inputs = extract(
            samples_data,
            "observations", "advantages", "pdists", "actions",
        )
        # Need to ensure this
        logger.log("computing loss before")
        loss_before = f_loss(*inputs)
        logger.log("performing update")
        # direct approach, just bite the bullet and use hessian
        _, flat_g, fisher_mat = f_fisher(*inputs)
        nat_direction = np.linalg.lstsq(fisher_mat, flat_g)

        # CG approach
        _, flat_g = f_grad(*inputs)
        def Hx(x):
            xs = policy.flat_to_params(x)
            with Message("rop"):
                rop = f_Hx_rop(*(inputs + xs))
            with Message("plain"):
                plain = f_Hx_plain(*(inputs + xs))
            assert np.allclose(rop, plain)
            return plain
            # alternatively we can do finite difference on flat_grad
        nat_direction_cg = cg(Hx, flat_g)

        logger.log("exact, cg direction diff %s" % (np.linalg.norm(nat_direction - nat_direction_cg)))
        nat_step_size = 1. if self.step_size is None \
            else (self.step_size * (
                1. / flat_g.T.dot(nat_direction)
            )) ** 0.5
        flat_descent_step = nat_step_size * nat_direction
        descent_steps = policy.flat_to_params(flat_descent_step)
        f_update(*descent_steps)
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
