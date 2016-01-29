from contextlib import contextmanager
from itertools import izip

import theano
import theano.tensor as TT
import numpy as np
from numpy.linalg import LinAlgError

from rllab.misc import logger, autoargs
from rllab.misc.krylov import cg
from rllab.misc.ext import extract, compile_function, flatten_hessian, new_tensor, new_tensor_like, \
    flatten_tensor_variables, lazydict, cached_function


class RecurrentNaturalGradientMethod(object):
    """
    Natural Gradient Method infrastructure for RNPG & RTRPO. (and possibly more)
    """

    @autoargs.arg("step_size", type=float,
                  help="KL divergence constraint. When it's None"
                       "only natural gradient direction is calculated")
    @autoargs.arg("use_cg", type=bool,
                  help="Directly estimate descent direction instead of inverting Fisher"
                       "Information matrix")
    @autoargs.arg("cg_iters", type=int,
                  help="The number of CG iterations used to calculate H^-1 g")
    @autoargs.arg("reg_coeff", type=float,
                  help="A small value to add to Fisher Information Matrix's eigenvalue"
                       "When CG is used, this value will not be changed but if we are"
                       "directly using Hessian inverse method, this regularization will be"
                       "adaptively increased should the regularized matrix is still singular"
                       "(but it's unlikely)")
    def __init__(
            self,
            step_size=0.001,
            use_cg=True,
            cg_iters=10,
            reg_coeff=1e-5,
            **kwargs):
        self.cg_iters = cg_iters
        self.use_cg = use_cg
        self.step_size = step_size
        self.reg_coeff = reg_coeff

    def init_opt(self, mdp, policy, baseline):
        obs_var = new_tensor(
            'input',
            ndim=2+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )
        advantage_var = TT.matrix('advantage')
        valid_var = TT.matrix('valid')
        action_var = new_tensor('action', ndim=3, dtype=mdp.action_dtype)
        log_prob = policy.get_log_prob_sym(obs_var, action_var)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        surr_obj = - TT.sum(log_prob * advantage_var * valid_var) / TT.sum(valid_var)
        # We would need to calculate the empirical fisher information matrix
        # This can be done as follows (though probably not the most efficient
        # way):
        # Since I(theta) = d^2 KL(p(theta)||p(theta')) / d theta^2
        #                  (evaluated at theta' = theta),
        # we can get I(theta) by calculating the hessian of
        # KL(p(theta)||p(theta'))
        old_pdist_var = new_tensor('old_pdist', ndim=3, dtype=theano.config.floatX)
        pdist_var = policy.get_pdist_sym(obs_var, action_var)
        mean_kl = TT.sum(valid_var * policy.kl(old_pdist_var, pdist_var)) / TT.sum(valid_var)
        grads = theano.grad(surr_obj, wrt=policy.get_params(trainable=True))
        flat_grad = flatten_tensor_variables(grads)

        logger.log("computing symbolic kl_grad")
        kl_grads = theano.grad(mean_kl, wrt=policy.get_params(trainable=True))
        # logger.log("computing symbolic empirical Fisher")
        # kl_flat_grad = flatten_tensor_variables(kl_grads)
        # emp_fisher = 
        xs = [
            new_tensor_like("%s x" % p.name, p)
            for p in policy.get_params(trainable=True)
            ]
        # many Ops don't have Rop implemented so this is not that useful
        # but the code implenting that is preserved for future reference
        # Hx_rop = TT.sum(TT.Rop(kl_flat_grad, policy.params, xs), axis=0)
        logger.log("computing symbolic Hessian vector product")
        Hx_plain_splits = TT.grad(TT.sum([
                                             TT.sum(g * x) for g, x in izip(kl_grads, xs)
                                             ]), wrt=policy.get_params(trainable=True))
        Hx_plain = TT.concatenate([s.flatten() for s in Hx_plain_splits])

        input_list = [
            obs_var, advantage_var, old_pdist_var, action_var, valid_var]
        f_loss = compile_function(
            inputs=input_list,
            outputs=surr_obj,
            log_name="f_loss",
        )
        f_grad = compile_function(
            inputs=input_list,
            outputs=[surr_obj, flat_grad],
            log_name="f_grad",
        )

        # follwoing information is computed to TRPO
        max_kl = TT.max(valid_var * policy.kl(old_pdist_var, pdist_var))

        return lazydict(
            f_loss=lambda: f_loss,
            f_grad=lambda: f_grad,
            f_fisher=lambda:
            compile_function(
                inputs=input_list,
                outputs=[
                    surr_obj,
                    flat_grad,
                    # This last term is the empirical Fisher information
                    flatten_hessian(mean_kl, wrt=policy.get_params(trainable=True), block_diagonal=False)
                ],
                log_name="f_fisher",
            ),
            f_Hx_plain=lambda:
            compile_function(
                inputs=input_list + xs,
                outputs=Hx_plain,
                log_name="f_Hx_plain",
            ),
            f_trpo_info=lambda:
            compile_function(
                inputs=input_list,
                outputs=[surr_obj, mean_kl, max_kl],
                log_name="f_trpo_info"
            ),
        )

    @contextmanager
    def optimization_setup(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        inputs = list(extract(
            samples_data,
            "observations", "advantages", "pdists", "actions", "valids"
        ))
        # Need to ensure this
        logger.log("computing loss before")
        loss_before = opt_info["f_loss"](*inputs)
        logger.log("performing update")
        logger.log("computing descent direction")
        if not self.use_cg:
            # direct approach, just bite the bullet and use hessian
            _, flat_g, fisher_mat = opt_info["f_fisher"](*inputs)
            while True:
                try:
                    nat_direction = np.linalg.solve(
                        fisher_mat + self.reg_coeff*np.eye(fisher_mat.shape[0]), flat_g
                    )
                    break
                except LinAlgError:
                    self.reg_coeff *= 5
                    print self.reg_coeff
        else:
            # CG approach
            _, flat_g = opt_info["f_grad"](*inputs)
            def Hx(x):
                xs = policy.flat_to_params(x, trainable=True)
                # with Message("rop"):
                #     rop = f_Hx_rop(*(inputs + xs))
                plain = opt_info["f_Hx_plain"](*(inputs + xs)) + self.reg_coeff*x
                # assert np.allclose(rop, plain)
                return plain
                # alternatively we can do finite difference on flat_grad
            nat_direction = cg(Hx, flat_g, cg_iters=self.cg_iters)

        nat_step_size = 1. if self.step_size is None \
            else (self.step_size * (
            1. / flat_g.T.dot(nat_direction)
        )) ** 0.5
        flat_descent_step = nat_step_size * nat_direction
        logger.log("descent direction computed")
        yield inputs, flat_descent_step
        logger.log("computing loss after")
        loss_after = opt_info["f_loss"](*inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.log("optimization finished")

    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data, opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )

