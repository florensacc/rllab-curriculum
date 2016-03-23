from contextlib import contextmanager
from itertools import izip

import theano
import theano.tensor as TT
import numpy as np
from numpy.linalg import LinAlgError

from rllab.misc import logger, autoargs
from rllab.misc.krylov import cg
from rllab.misc import ext


class NaturalGradientMethod(object):
    """
    Natural Gradient Method infrastructure for NPG & TRPO. (and possibly more)
    """

    def __init__(
            self,
            step_size=0.001,
            use_cg=True,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=0.1,
            trpo_stepsize=False,
            maybe_aggressive=False,
            **kwargs):
        """

        :param step_size: KL divergence constraint. When it's None only natural gradient direction is calculated.
        :param use_cg: Directly estimate descent direction instead of inverting Fisher Information matrix
        :param cg_iters: The number of CG iterations used to calculate H^-1 g
        :param reg_coeff: A small value to add to Fisher Information Matrix's eigenvalue. When CG is used, this value
        will not be changed but if we are directly using Hessian inverse method, this regularization will be
        adaptively increased should the regularized matrix still be singular (but it's unlikely)
        :param subsample_factor: Subsampling factor to reduce samples when using conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param trpo_stepsize: whether to use stepsize for trpo
        :param maybe_aggressive: whether to be aggressive in descent direction estimate
        :param kwargs:
        :return:
        """
        self.maybe_aggressive = maybe_aggressive
        self.trpo_stepsize = trpo_stepsize
        self.cg_iters = cg_iters
        self.use_cg = use_cg
        self.step_size = step_size
        self.reg_coeff = reg_coeff
        self.subsample_factor = subsample_factor

    def init_opt(self, env_spec, policy, baseline):
        is_recurrent = int(policy.recurrent)
        obs_var = env_spec.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        advantage_var = ext.new_tensor(
            'advantage',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX
        )
        action_var = env_spec.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )

        old_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in policy.dist_info_keys
            }
        old_info_vars_list = [old_info_vars[k] for k in policy.dist_info_keys]

        if is_recurrent:
            valid_var = TT.matrix('valid')
        else:
            valid_var = None
        info_vars = policy.dist_info_sym(obs_var, action_var)
        if is_recurrent:
            mean_kl = TT.sum(policy.kl_sym(old_info_vars, info_vars) * valid_var) / TT.sum(valid_var)
        else:
            mean_kl = TT.mean(policy.kl_sym(old_info_vars, info_vars))
        lr = policy.likelihood_ratio_sym(action_var, old_info_vars, info_vars)
        # formulate as a minimization problem
        # The gradient of the surrogate objective is the policy gradient
        if is_recurrent:
            surr_obj = - TT.sum(lr * advantage_var * valid_var) / TT.sum(valid_var)
        else:
            surr_obj = - TT.mean(lr * advantage_var)
        # We would need to calculate the empirical fisher information matrix
        # This can be done as follows (though probably not the most efficient
        # way):
        # Since I(theta) = d^2 KL(p(theta)||p(theta')) / d theta^2
        #                  (evaluated at theta' = theta),
        # we can get I(theta) by calculating the hessian of
        # KL(p(theta)||p(theta'))
        grads = theano.grad(surr_obj, wrt=policy.get_params(trainable=True))
        flat_grad = ext.flatten_tensor_variables(grads)

        kl_grads = theano.grad(mean_kl, wrt=policy.get_params(trainable=True))
        # kl_flat_grad = flatten_tensor_variables(kl_grads)
        xs = [
            ext.new_tensor_like("%s x" % p.name, p)
            for p in policy.get_params(trainable=True)
            ]
        # many Ops don't have Rop implemented so this is not that useful
        # but the code implementing that is preserved for future reference
        # Hx_rop = TT.sum(TT.Rop(kl_flat_grad, policy.params, xs), axis=0)
        Hx_plain_splits = TT.grad(TT.sum([
                                             TT.sum(g * x) for g, x in izip(kl_grads, xs)
                                             ]), wrt=policy.get_params(trainable=True))
        Hx_plain = TT.concatenate([s.flatten() for s in Hx_plain_splits])

        input_list = [obs_var, action_var, advantage_var] + old_info_vars_list
        if is_recurrent:
            input_list.append(valid_var)

        # following information is computed to TRPO
        max_kl = TT.max(policy.kl_sym(old_info_vars, info_vars))

        return ext.lazydict(
            f_loss=lambda: ext.compile_function(
                inputs=input_list,
                outputs=surr_obj,
                log_name="f_loss",
            ),
            f_grad=lambda: ext.compile_function(
                inputs=input_list,
                outputs=[surr_obj, flat_grad],
                log_name="f_grad",
            ),
            f_fisher=lambda: ext.compile_function(
                inputs=input_list,
                outputs=[
                    surr_obj,
                    flat_grad,
                    ext.flatten_hessian(mean_kl, wrt=policy.get_params(trainable=True), block_diagonal=False)
                ],
                log_name="f_fisher",
            ),
            f_Hx_plain=lambda: ext.compile_function(
                inputs=input_list + xs,
                outputs=Hx_plain,
                log_name="f_Hx_plain",
            ),
            f_trpo_info=lambda: ext.compile_function(
                inputs=input_list,
                outputs=[surr_obj, mean_kl, max_kl],
                log_name="f_trpo_info"
            ),
        )

    @contextmanager
    def optimization_setup(self, itr, policy, samples_data, opt_info):
        logger.log("optimizing policy")
        inputs = list(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        info_list = [agent_infos[k] for k in policy.dist_info_keys]
        inputs.extend(info_list)
        if policy.recurrent:
            inputs.append(samples_data["valids"])
        if self.subsample_factor < 1:
            n_samples = len(inputs[0])
            inds = np.random.choice(
                n_samples, n_samples * self.subsample_factor, replace=False)
            subsample_inputs = [x[inds] for x in inputs]
        else:
            subsample_inputs = inputs
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
                        fisher_mat + self.reg_coeff * np.eye(fisher_mat.shape[0]), flat_g
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
                plain = opt_info["f_Hx_plain"](*(subsample_inputs + xs)) + self.reg_coeff * x
                # assert np.allclose(rop, plain)
                return plain
                # alternatively we can do finite difference on flat_grad

            nat_direction = cg(Hx, flat_g, cg_iters=self.cg_iters)

        nat_step_size = 1. if self.step_size is None \
            else ((2 if self.trpo_stepsize else 1) * self.step_size * (
            1. / (flat_g.T.dot(nat_direction) + 1e-8
                  if not self.maybe_aggressive
                  else nat_direction.dot(Hx(nat_direction)) + 1e-8)
        )) ** 0.5
        flat_descent_step = nat_step_size * nat_direction
        logger.log("descent direction computed")
        yield inputs, flat_descent_step
        logger.log("computing loss after")
        loss_after = opt_info["f_loss"](*inputs)
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.log("optimization finished")

    def get_itr_snapshot(self, itr, env, policy, baseline, samples_data, opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            env=env,
        )
