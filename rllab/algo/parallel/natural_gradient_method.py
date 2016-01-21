from contextlib import contextmanager
from itertools import izip

import theano
import theano.tensor as TT
import numpy as np
from numpy.linalg import LinAlgError

from rllab.algo.batch_polopt import BatchPolopt
from rllab.misc import logger, autoargs
from rllab.misc.krylov import cg
from rllab.sampler import parallel_sampler
from rllab.misc.ext import extract, compile_function, flatten_hessian, \
    new_tensor, new_tensor_like, flatten_tensor_variables, lazydict


PG = parallel_sampler.G


class Globals():

    def __init__(self):
        self.opt = None
        self.opt_info = None
        self.inputs = None
        self.subsample_inputs = None

G = Globals()


def worker_prepare_inputs():
    G.inputs = list(extract(
        PG.samples_data,
        "observations", "advantages", "pdists", "actions"
    ))
    if G.opt.subsample_factor < 1:
        n_samples = len(G.inputs[0])
        inds = np.random.choice(
            n_samples, n_samples * G.opt.subsample_factor, replace=False)
        G.subsample_inputs = [x[inds] for x in G.inputs]
    else:
        G.subsample_inputs = G.inputs


def worker_f(f_name, *args):
    if f_name == "Hx_plain" or f_name == "fisher":
        inputs = G.subsample_inputs
    else:
        inputs = G.inputs
    return G.opt_info[f_name](*(inputs + list(args)))


def master_f(f_name):
    def f(*args):
        return parallel_sampler.master_collect_mean(worker_f, f_name, *args)
    return f


def worker_init_opt(opt):
    mdp = PG.mdp
    policy = PG.policy
    G.opt = opt
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
    grads = theano.grad(surr_obj, wrt=policy.get_params(trainable=True))
    flat_grad = flatten_tensor_variables(grads)

    kl_grads = theano.grad(mean_kl, wrt=policy.get_params(trainable=True))
    # kl_flat_grad = flatten_tensor_variables(kl_grads)
    emp_fisher = flatten_hessian(
        mean_kl, wrt=policy.get_params(trainable=True), block_diagonal=False)
    xs = [
        new_tensor_like("%s x" % p.name, p)
        for p in policy.get_params(trainable=True)
        ]
    # many Ops don't have Rop implemented so this is not that useful
    # but the code implenting that is preserved for future reference
    # Hx_rop = TT.sum(TT.Rop(kl_flat_grad, policy.params, xs), axis=0)
    Hx_plain_splits = TT.grad(
        TT.sum([
            TT.sum(g * x) for g, x in izip(kl_grads, xs)
        ]), wrt=policy.get_params(trainable=True))
    Hx_plain = TT.concatenate([s.flatten() for s in Hx_plain_splits])

    input_list = [input_var, advantage_var, old_pdist_var, action_var]
    f_loss = compile_function(
        inputs=input_list,
        outputs=surr_obj,
    )
    f_grad = compile_function(
        inputs=input_list,
        outputs=[surr_obj, flat_grad],
    )

    # follwoing information is computed to TRPO
    max_kl = TT.max(policy.kl(old_pdist_var, pdist_var))

    G.opt_info = lazydict(
        f_loss=lambda: f_loss,
        f_grad=lambda: f_grad,
        f_fisher=lambda:
        compile_function(
            inputs=input_list,
            outputs=[surr_obj, flat_grad, emp_fisher],
        ),
        f_Hx_plain=lambda:
        compile_function(
            inputs=input_list + xs,
            outputs=Hx_plain,
        ),
        f_trpo_info=lambda:
        compile_function(
            inputs=input_list,
            outputs=[surr_obj, mean_kl, max_kl]
        ),
    )


class NaturalGradientMethod(BatchPolopt):
    """
    Natural Gradient Method infrastructure for NPG & TRPO. (and possibly more)
    """

    @autoargs.arg("step_size", type=float,
                  help="KL divergence constraint. When it's None"
                       "only natural gradient direction is calculated")
    @autoargs.arg("use_cg", type=bool,
                  help="Directly estimate descent direction instead of "
                       "inverting Fisher Information matrix")
    @autoargs.arg("cg_iters", type=int,
                  help="The number of CG iterations used to calculate H^-1 g")
    @autoargs.arg("reg_coeff", type=float,
                  help="A small value to add to Fisher Information Matrix's "
                       "eigenvalue. When CG is used, this value will not be "
                       "changed but if we are directly using Hessian inverse "
                       "method, this regularization will be adaptively "
                       "increased should the regularized matrix is still "
                       "singular (but it's unlikely)")
    @autoargs.arg("subsample_factor", type=float,
                  help="Subsampling factor to reduce samples when using "
                       "conjugate gradient. Since the computation time for "
                       "the descent direction dominates, this can greatly "
                       "reduce the overall computation time.")
    def __init__(
            self,
            step_size=0.001,
            use_cg=True,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=0.1,
            **kwargs):
        self.opt.cg_iters = cg_iters
        self.opt.use_cg = use_cg
        self.opt.step_size = step_size
        self.opt.reg_coeff = reg_coeff
        self.opt.subsample_factor = subsample_factor

    def init_opt(self, mdp, policy, baseline):
        parallel_sampler.run_map(worker_init_opt, self.opt)

    @contextmanager
    def optimization_setup(self, itr, policy, samples_data, opt_info):
        parallel_sampler.run_map(worker_prepare_inputs)
        logger.log("optimizing policy")
        logger.log("computing loss before")
        loss_before = master_f("f_loss")()
        logger.log("performing update")
        logger.log("computing descent direction")
        if not self.opt.use_cg:
            # direct approach, just bite the bullet and use hessian
            _, flat_g, fisher_mat = master_f("f_fisher")()
            while True:
                reg_fisher_mat = fisher_mat + \
                    self.opt.reg_coeff*np.eye(fisher_mat.shape[0])
                try:
                    nat_direction = np.linalg.solve(
                        reg_fisher_mat, flat_g
                    )
                    break
                except LinAlgError:
                    self.opt.reg_coeff *= 5
                    print self.opt.reg_coeff
        else:
            # CG approach
            _, flat_g = master_f("f_grad")()

            def Hx(x):
                xs = policy.flat_to_params(x)
                # with Message("rop"):
                #     rop = f_Hx_rop(*(inputs + xs))
                plain = master_f("f_Hx_plain")(*xs) + self.opt.reg_coeff*x
                # assert np.allclose(rop, plain)
                return plain
                # alternatively we can do finite difference on flat_grad
            nat_direction = cg(Hx, flat_g, cg_iters=self.opt.cg_iters)

        nat_step_size = 1. if self.opt.step_size is None \
            else (self.opt.step_size * (
                1. / flat_g.T.dot(nat_direction)
            )) ** 0.5
        flat_descent_step = nat_step_size * nat_direction
        logger.log("descent direction computed")
        yield flat_descent_step
        logger.log("computing loss after")
        loss_after = master_f("f_loss")()
        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.log("optimization finished")

    def get_itr_snapshot(
            self, itr, mdp, policy, baseline, samples_data, opt_info):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )
