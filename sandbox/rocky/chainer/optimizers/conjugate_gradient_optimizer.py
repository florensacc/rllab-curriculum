from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import numpy as np
from sandbox.rocky.chainer.misc import tensor_utils


class FiniteDifferenceHvp(object):
    def __init__(self, eps=1e-5, symmetric=True, grad_clip=None, num_slices=1):
        self.eps = eps
        self.symmetric = symmetric
        self.grad_clip = grad_clip
        self.num_slices = num_slices
        self.target = None
        self.reg_coeff = None
        self.f_loss_constraint = None

    def update_opt(self, target, f_loss_constraint, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        self.f_loss_constraint = f_loss_constraint

    def build_eval(self, inputs):
        def eval(x):
            def f_grad(flat_param):
                self.target.set_param_values(flat_param, trainable=True)
                self.target.zerograds()
                _, constr = tensor_utils.sliced_fun_sym(self.f_loss_constraint, self.num_slices)(inputs, ())
                constr.backward()
                return self.target.get_grad_values(trainable=True)

            param_val = self.target.get_param_values(trainable=True)
            flat_grad_dvplus = f_grad(param_val + self.eps * x)
            if self.symmetric:
                flat_grad_dvminus = f_grad(param_val - self.eps * x)
                hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * self.eps)
                self.target.set_param_values(param_val, trainable=True)
            else:
                flat_grad = f_grad(param_val)
                hx = (flat_grad_dvplus - flat_grad) / self.eps
            return hx + self.reg_coeff * x

        return eval


class ConjugateGradientOptimizer(Serializable):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            accept_violation=False,
            hvp_approach=None,
            num_slices=1):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param accept_violation: whether to accept the descent step if it violates the line search condition after
        exhausting all backtracking budgets
        :return:
        """
        Serializable.quick_init(self, locals())
        self.cg_iters = cg_iters
        self.reg_coeff = reg_coeff
        self.subsample_factor = subsample_factor
        self.backtrack_ratio = backtrack_ratio
        self.max_backtracks = max_backtracks
        self.num_slices = num_slices

        self.opt_fun = None
        self.target = None
        self.max_constraint_val = None
        self.constraint_name = None
        self.accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = FiniteDifferenceHvp(num_slices)
        self.hvp_approach = hvp_approach

        self.target = None
        self.f_loss_constraint = None
        self.max_constraint_val = None

    def update_opt(self, target, f_loss_constraint, max_constraint_val):
        self.target = target
        self.f_loss_constraint = f_loss_constraint
        self.max_constraint_val = max_constraint_val
        self.hvp_approach.update_opt(
            target=target,
            f_loss_constraint=f_loss_constraint,
            reg_coeff=self.reg_coeff
        )

    def loss_constraint(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        loss, constr = tensor_utils.sliced_fun_sym(self.f_loss_constraint, self.num_slices)(inputs, extra_inputs)
        return loss.data, constr.data

    def optimize(self, inputs, extra_inputs=None, subsample_grouped_inputs=None):
        prev_param = np.copy(self.target.get_param_values(trainable=True))
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self.subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self.subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("Start CG optimization: #parameters: %d, #inputs: %d, #subsample_inputs: %d" % (
            len(prev_param), len(inputs[0]), len(subsample_inputs[0])))

        self.target.zerograds()

        logger.log("computing loss before")
        loss_before, _ = tensor_utils.sliced_fun_sym(
            self.f_loss_constraint,
            self.num_slices
        )(inputs, extra_inputs)
        logger.log("performing update")

        logger.log("computing gradient")
        loss_before.backward()
        flat_g = self.target.get_grad_values(trainable=True)

        logger.log("gradient computed")

        logger.log("computing descent direction")
        Hx = self.hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self.cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self.max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        n_iter = 0
        for n_iter, ratio in enumerate(self.backtrack_ratio ** np.arange(self.max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self.target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = tensor_utils.sliced_fun_sym(
                self.f_loss_constraint,
                self.num_slices
            )(inputs, extra_inputs)
            if loss.data < loss_before.data and constraint_val.data <= self.max_constraint_val:
                break
        if (np.isnan(loss.data) or np.isnan(constraint_val.data) or loss.data >= loss_before.data or
                    constraint_val.data >= self.max_constraint_val) and not self.accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" % self.constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self.max_constraint_val:
                logger.log("Violated because constraint %s is violated" % self.constraint_name)
            self.target.set_param_values(prev_param, trainable=True)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")
