from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np
from rllab.misc.ext import sliced_fun
from _ast import Num


# class PerlmutterHvp(Serializable):
#
#     def __init__(self, num_slices=1):
#         Serializable.quick_init(self, locals())
#         self.target = None
#         self.reg_coeff = None
#         self.opt_fun = None
#         self._num_slices = num_slices
#
#     def update_opt(self, f0, target, inputs, reg_coeff, f1=None, f2=None,):
#         self.target = target
#         self.reg_coeff = reg_coeff
#         params = target.get_params(trainable=True)
#
#         constraint_grads0 = theano.grad(
#             f0, wrt=params, disconnected_inputs='warn')
#         if f1 is not None:
#             constraint_grads1 = theano.grad(
#                 f1, wrt=params, disconnected_inputs='warn')
#
#         if f2 is not None:
#             constraint_grads2 = theano.grad(
#                 f2, wrt=params, disconnected_inputs='warn')
#
#         xs = tuple([ext.new_tensor_like("%s x" % p.name, p) for p in params])
#
#         def Hx_sum():
#             Hx_sum_splits = TT.grad(
#                 TT.sum([TT.sum(g * x)
#                         for g, x in zip(constraint_grads0, xs)]),
#                 wrt=params,
#                 disconnected_inputs='warn'
#             )
#             if f1 is not None:
#                 Hx_sum_splits += TT.grad(
#                     TT.sum([TT.sum(g * x)
#                             for g, x in zip(constraint_grads1, xs)]),
#                     wrt=params,
#                     disconnected_inputs='warn'
#                 )
#             if f2 is not None:
#                 Hx_sum_splits += TT.grad(
#                     TT.sum([TT.sum(g * x)
#                             for g, x in zip(constraint_grads2, xs)]),
#                     wrt=params,
#                     disconnected_inputs='warn'
#                 )
#             return TT.concatenate([TT.flatten(s) for s in Hx_sum_splits])
#
#         def Hx_0():
#             Hx_0_splits = TT.grad(
#                 TT.sum([TT.sum(g * x)
#                         for g, x in zip(constraint_grads0, xs)]),
#                 wrt=params,
#                 disconnected_inputs='warn'
#             )
#             return TT.concatenate([TT.flatten(s) for s in Hx_0_splits])
#
#         def Hx_1():
#             Hx_1_splits = TT.grad(
#                 TT.sum([TT.sum(g * x)
#                         for g, x in zip(constraint_grads1, xs)]),
#                 wrt=params,
#                 disconnected_inputs='warn'
#             )
#             return TT.concatenate([TT.flatten(s) for s in Hx_1_splits])
#
#         self.opt_fun = ext.lazydict(
#             f_Hx_sum=lambda: ext.compile_function(
#                 inputs=inputs + xs,
#                 outputs=Hx_sum(),
#                 log_name="f_Hx_sum",
#             ),
#             f_Hx_0=lambda: ext.compile_function(
#                 inputs=inputs + xs,
#                 outputs=Hx_0(),
#                 log_name="f_Hx_0",
#             ),
#             f_Hx_1=lambda: ext.compile_function(
#                 inputs=inputs + xs,
#                 outputs=Hx_1(),
#                 log_name="f_Hx_1",
#             )
#         )
#
#     def build_eval_sum(self, inputs):
#         def eval(x):
#             xs = tuple(self.target.flat_to_params(x, trainable=True))
#             ret = sliced_fun(self.opt_fun["f_Hx_sum"], self._num_slices)(
#                 inputs, xs) + self.reg_coeff * x
#             return ret
#
#         return eval
#
#     def build_eval_0(self, inputs):
#         def eval(x):
#             xs = tuple(self.target.flat_to_params(x, trainable=True))
#             ret = sliced_fun(self.opt_fun["f_Hx_0"], self._num_slices)(
#                 inputs, xs) + self.reg_coeff * x
#             return ret
#
#         return eval
#
#     def build_eval_1(self, inputs):
#         def eval(x):
#             xs = tuple(self.target.flat_to_params(x, trainable=True))
#             ret = sliced_fun(self.opt_fun["f_Hx_1"], self._num_slices)(
#                 inputs, xs) + self.reg_coeff * x
#             return ret
#
#         return eval

class PerlmutterHvp(Serializable):

    def __init__(self, num_slices=1):
        Serializable.quick_init(self, locals())
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff
        params = target.get_params(trainable=True)

        constraint_grads = theano.grad(
            f, wrt=params, disconnected_inputs='warn')
        xs = tuple([ext.new_tensor_like("%s x" % p.name, p) for p in params])

        def Hx_plain():
            Hx_plain_splits = TT.grad(
                TT.sum([TT.sum(g * x)
                        for g, x in zip(constraint_grads, xs)]),
                wrt=params,
                disconnected_inputs='warn'
            )
            return TT.concatenate([TT.flatten(s) for s in Hx_plain_splits])

        self.opt_fun = ext.lazydict(
            f_Hx_plain=lambda: ext.compile_function(
                inputs=inputs + xs,
                outputs=Hx_plain(),
                log_name="f_Hx_plain",
            ),
        )

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(
                inputs, xs) + self.reg_coeff * x
            return ret

        return eval



class FiniteDifferenceHvp(Serializable):

    def __init__(self, base_eps=1e-8, symmetric=True, grad_clip=None, num_slices=1):
        Serializable.quick_init(self, locals())
        self.base_eps = base_eps
        self.symmetric = symmetric
        self.grad_clip = grad_clip
        self._num_slices = num_slices

    def update_opt(self, f, target, inputs, reg_coeff):
        self.target = target
        self.reg_coeff = reg_coeff

        params = target.get_params(trainable=True)

        constraint_grads = theano.grad(
            f, wrt=params, disconnected_inputs='warn')
        flat_grad = ext.flatten_tensor_variables(constraint_grads)

        def f_Hx_plain(*args):
            inputs_ = args[:len(inputs)]
            xs = args[len(inputs):]
            flat_xs = np.concatenate([np.reshape(x, (-1,)) for x in xs])
            param_val = self.target.get_param_values(trainable=True)
            eps = np.cast['float32'](
                self.base_eps / (np.linalg.norm(param_val) + 1e-8))
            self.target.set_param_values(
                param_val + eps * flat_xs, trainable=True)
            flat_grad_dvplus = self.opt_fun["f_grad"](*inputs_)
            if self.symmetric:
                self.target.set_param_values(
                    param_val - eps * flat_xs, trainable=True)
                flat_grad_dvminus = self.opt_fun["f_grad"](*inputs_)
                hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * eps)
                self.target.set_param_values(param_val, trainable=True)
            else:
                self.target.set_param_values(param_val, trainable=True)
                flat_grad = self.opt_fun["f_grad"](*inputs_)
                hx = (flat_grad_dvplus - flat_grad) / eps
            return hx

        self.opt_fun = ext.lazydict(
            f_grad=lambda: ext.compile_function(
                inputs=inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_Hx_plain=lambda: f_Hx_plain,
        )

    def build_eval(self, inputs):
        def eval(x):
            xs = tuple(self.target.flat_to_params(x, trainable=True))
            ret = sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(
                inputs, xs) + self.reg_coeff * x
            return ret

        return eval


class ConjugateGradientOptimizer2(Serializable):
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
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks
        self._num_slices = num_slices

        self._opt_fun = None
        self._target = None
        self._max_kl_constraint_val = None
        self._kl_constraint_name = None
        self._max_entropy_constraint_val = None
        self._entropy_term_name = None
        self._max_performance_constraint_val = None
        self._performance_constraint_name = None
        self._kl_direction = True
        self._entropy_direction = True
        self._performance_direction = False
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

    def update_opt(self, loss, target,
                   kl_constraint,
                   entropy_term,
                   performance_constraint,
                   inputs, extra_inputs=None,
                   constraint_names=("kl_constraint","entropy_constraint", "performance_constraint"),
                   *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraints: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs, which could be subsampled if needed. It is assumed
        that the first dimension of these inputs should correspond to the number of data points
        :param extra_inputs: A list of symbolic variables as extra inputs which should not be subsampled
        :return: No return value.
        """

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        else:
            extra_inputs = tuple(extra_inputs)

        kl_constraint_term, kl_constraint_value = kl_constraint
        performance_constraint_term, performance_constraint_value = performance_constraint

        params = target.get_params(trainable=True)
        loss_grads = theano.grad(loss, wrt=params, disconnected_inputs='warn')
        entropy_grads = theano.grad(entropy_term, wrt=params, disconnected_inputs='warn')
        performance_grads = theano.grad(performance_constraint_term, wrt=params, disconnected_inputs='warn')

        # grads = theano.grad(loss, wrt=params, disconnected_inputs='warn') - \
        #         theano.grad(entropy_constraint_term, wrt=params, disconnected_inputs='warn') - \
        #         theano.grad(performance_constraint_term, wrt=params, disconnected_inputs='warn')
        flat_loss_grad = ext.flatten_tensor_variables(loss_grads)
        flat_entropy_grad = ext.flatten_tensor_variables(entropy_grads)
        flat_performance_grad = ext.flatten_tensor_variables(performance_grads)
        flat_grad = flat_loss_grad + flat_entropy_grad + flat_performance_grad

        self._hvp_approach.update_opt(f=kl_constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        self._max_kl_constraint_val = kl_constraint_value
        self._kl_constraint_name = constraint_names[0]
        self._entropy_term_name = constraint_names[1]
        self._max_performance_constraint_val = performance_constraint_value
        self._performance_constraint_name = constraint_names[2]

        self._opt_fun = ext.lazydict(
            f_loss=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
            ),
            f_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_kl_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=kl_constraint_term,
                log_name="kl_constraint",
            ),
            f_entropy_term=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=entropy_term,
                log_name="entropy_term",
            ),
            f_performance_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=performance_constraint_term,
                log_name="performance_constraint",
            ),
            f_loss_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, kl_constraint_term, entropy_term, performance_constraint_term],
                log_name="f_loss_constraint",
            ),
        )

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)

    def constraint_val(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return (sliced_fun(self._opt_fun["f_kl_constraint"], self._num_slices)(inputs, extra_inputs),
                sliced_fun(self._opt_fun["f_entropy_term"], self._num_slices)(inputs, extra_inputs),
                sliced_fun(self._opt_fun["f_performance_constraint"], self._num_slices)(inputs, extra_inputs))

    def optimize(self, inputs, extra_inputs=None, subsample_grouped_inputs=None):

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self._subsample_factor < 1:
            if subsample_grouped_inputs is None:
                subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
        else:
            subsample_inputs = inputs

        logger.log("computing loss before")
        loss_before = sliced_fun(self._opt_fun["f_loss"], self._num_slices)(
            inputs, extra_inputs)
        entropy_term_before = sliced_fun(self._opt_fun["f_entropy_term"], self._num_slices)(
            inputs, extra_inputs)
        logger.log("performing update")
        logger.log("computing descent direction")

        flat_g = sliced_fun(self._opt_fun["f_grad"], self._num_slices)(
            inputs, extra_inputs)

        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)
        # descent_direction /= np.linalg.norm(descent_direction)
        initial_step_size = np.sqrt(
            2 * self._max_kl_constraint_val *
            (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        if np.isnan(initial_step_size):
            print("WARNING!!!!!")
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        prev_param = np.copy(self._target.get_param_values(trainable=True))
        n_iter = 0
        # loss, kl_constraint_val, entropy_term, performance_constraint_val = sliced_fun(
        #     self._opt_fun["f_loss_constraint"], self._num_slices)(inputs, extra_inputs)
        # print(kl_constraint_val)
        # print("KLKLKLKL")
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)
            loss, kl_constraint_val, entropy_term, performance_constraint_val = sliced_fun(
                self._opt_fun["f_loss_constraint"], self._num_slices)(inputs, extra_inputs)
            print(kl_constraint_val)
            if loss < loss_before and kl_constraint_val <= self._max_kl_constraint_val and \
                    entropy_term <= entropy_term_before and \
                    performance_constraint_val <= self._max_performance_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(kl_constraint_val) or
                np.isnan(entropy_term) or np.isnan(performance_constraint_val) or loss >= loss_before or
                kl_constraint_val >= self._max_kl_constraint_val or
                entropy_term >= entropy_term_before or
                performance_constraint_val >= self._max_performance_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(kl_constraint_val):
                logger.log("Violated because constraint %s is NaN" %
                           self._kl_constraint_name)
            if np.isnan(entropy_term):
                logger.log("Violated because constraint %s is NaN" %
                           self._entropy_term_name)
            if np.isnan(performance_constraint_val):
                logger.log("Violated because constraint %s is NaN" %
                           self._performance_constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if kl_constraint_val >= self._max_kl_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" % self._kl_constraint_name)
            if entropy_term >= entropy_term_before:
                logger.log(
                    "Violated because constraint %s is violated" % self._entropy_term_name)
            if performance_constraint_val >= self._max_performance_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" % self._performance_constraint_name)
            self._target.set_param_values(prev_param, trainable=True)
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")
