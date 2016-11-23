from rllab.misc import ext
# from rllab.misc import krylov
from sandbox.adam.gpar.optimizers import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
# import itertools
import numpy as np
from rllab.misc.ext import sliced_fun
# from _ast import Num
from sandbox.adam.util import struct
import multiprocessing as mp
from ctypes import c_bool


class ParallelPerlmutterHvp(Serializable):

    def __init__(self, num_slices=1):
        Serializable.quick_init(self, locals())
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices

    def __getstate__(self):
        """ Do not pickle parallel objects """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

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
        def parallel_eval(x):
            shareds = self.par_objs.shareds
            vb = self.par_data.vb_pair

            xs = tuple(self.target.flat_to_params(x, trainable=True))

            shareds.grads_2d[self.par_data.rank, :] = self.par_data.avg_fac * \
                sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(inputs, xs)
            self.par_objs.barrier[0].wait()

            shareds.Hx[vb[0]:vb[1]] = self.reg_coeff * x[vb[0]:vb[1]] + \
                shareds.grads_2d[:, vb[0]:vb[1]].sum(axis=0)
            self.par_objs.barrier[1].wait()
            return shareds.Hx  # (or access this elsewhere)

        return parallel_eval


# class FiniteDifferenceHvp(Serializable):

#     def __init__(self, base_eps=1e-8, symmetric=True, grad_clip=None, num_slices=1):
#         Serializable.quick_init(self, locals())
#         self.base_eps = base_eps
#         self.symmetric = symmetric
#         self.grad_clip = grad_clip
#         self._num_slices = num_slices

#     def update_opt(self, f, target, inputs, reg_coeff):
#         self.target = target
#         self.reg_coeff = reg_coeff

#         params = target.get_params(trainable=True)

#         constraint_grads = theano.grad(
#             f, wrt=params, disconnected_inputs='warn')
#         flat_grad = ext.flatten_tensor_variables(constraint_grads)

#         def f_Hx_plain(*args):
#             inputs_ = args[:len(inputs)]
#             xs = args[len(inputs):]
#             flat_xs = np.concatenate([np.reshape(x, (-1,)) for x in xs])
#             param_val = self.target.get_param_values(trainable=True)
#             eps = np.cast['float32'](
#                 self.base_eps / (np.linalg.norm(param_val) + 1e-8))
#             self.target.set_param_values(
#                 param_val + eps * flat_xs, trainable=True)
#             flat_grad_dvplus = self.opt_fun["f_grad"](*inputs_)
#             if self.symmetric:
#                 self.target.set_param_values(
#                     param_val - eps * flat_xs, trainable=True)
#                 flat_grad_dvminus = self.opt_fun["f_grad"](*inputs_)
#                 hx = (flat_grad_dvplus - flat_grad_dvminus) / (2 * eps)
#                 self.target.set_param_values(param_val, trainable=True)
#             else:
#                 self.target.set_param_values(param_val, trainable=True)
#                 flat_grad = self.opt_fun["f_grad"](*inputs_)
#                 hx = (flat_grad_dvplus - flat_grad) / eps
#             return hx

#         self.opt_fun = ext.lazydict(
#             f_grad=lambda: ext.compile_function(
#                 inputs=inputs,
#                 outputs=flat_grad,
#                 log_name="f_grad",
#             ),
#             f_Hx_plain=lambda: f_Hx_plain,
#         )

#     def build_eval(self, inputs):
#         def eval(x):
#             xs = tuple(self.target.flat_to_params(x, trainable=True))
#             ret = sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(
#                 inputs, xs) + self.reg_coeff * x
#             return ret

#         return eval


class ParallelConjugateGradientOptimizer(Serializable):
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
        self._max_constraint_val = None
        self._constraint_name = None
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = ParallelPerlmutterHvp(num_slices)
        self._hvp_approach = hvp_approach

    def __getstate__(self):
        """ Do not pickle parallel objects """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def initialize_par_objs(self, n_parallel, size_grad):
        """ Before forking subprocesses """
        n = n_parallel
        self.n_paralllel = n
        n_elm_per_worker = -(-size_grad // n)  # (ceiling div)
        vb_idx = [n_elm_per_worker * i for i in range(n + 1)]
        vb_idx[-1] = size_grad
        vb_pairs = [(vb_idx[i], vb_idx[i + 1]) for i in range(n)]

        self.floatX = theano.config.floatX
        if self.floatX == 'float32':
            tc = 'f'  # typecode for single precision: c_float
        elif self.floatX == 'float64':
            tc = 'd'  # typecode for double precision: c_double
        else:
            raise ValueError("Theano floatX unsupported (only 32 or 64): ", self.floatX)

        par_data = struct(
            rank=None,
            avg_fac=np.array(1.0 / n, dtype=self.floatX),
            vb_pairs=vb_pairs)
        self.par_data = par_data
        self._hvp_approach.par_data = par_data

        n_arr = np.ctypeslib.as_array  # shorthand
        m_arr = mp.RawArray
        shareds = struct(
            flat_g=n_arr(m_arr(tc, size_grad)),
            grads_2d=n_arr(m_arr(tc, n * size_grad)).reshape(n, size_grad),
            Hx=n_arr(m_arr(tc, size_grad)),
            loss=n_arr(m_arr(tc, n)),
            constraint_val=n_arr(m_arr(tc, n)),
            descent=n_arr(m_arr(tc, size_grad)),
            prev_param=n_arr(m_arr(tc, size_grad)),
            cur_param=n_arr(m_arr(tc, size_grad)),
            n_samples=m_arr('i', n),
        )
        barriers = struct(
            avg_fac=mp.Barrier(n),
            flat_g=[mp.Barrier(n) for _ in range(2)],
            loss_cnstr=mp.Barrier(n),
            bktrk=mp.Barrier(n),
        )
        hvp_shareds = struct(
            grads_2d=shareds.grads_2d,  # OK to use the same memory
            Hx=shareds.Hx,
        )
        self._hvp_approach.par_objs = struct(
            shareds=hvp_shareds,
            barrier=[mp.Barrier(n) for _ in range(2)]
        )
        cg_par_objs = struct(
            z=shareds.Hx,  # (location of result of Hx)
            p=n_arr(m_arr(tc, size_grad)),
            x=shareds.descent,  # (location to write krylov.cg result)
            brk=mp.RawValue(c_bool, False),
            barrier=mp.Barrier(n),
        )
        self.par_objs = struct(
            shareds=shareds,
            barriers=barriers,
            cg_par_objs=cg_par_objs,
        )

    def initialize_rank(self, rank):
        self.par_data.rank = rank
        self.par_data.vb_pair = self.par_data.vb_pairs[rank]

    def update_opt(self, loss, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint", *args,
                   **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
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

        constraint_term, constraint_value = leq_constraint

        params = target.get_params(trainable=True)
        grads = theano.grad(loss, wrt=params, disconnected_inputs='warn')
        flat_grad = ext.flatten_tensor_variables(grads)

        self._hvp_approach.update_opt(f=constraint_term, target=target, inputs=inputs + extra_inputs,
                                      reg_coeff=self._reg_coeff)

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._opt_fun = ext.lazydict(
            # f_loss=lambda: ext.compile_function(
            #     inputs=inputs + extra_inputs,
            #     outputs=loss,
            #     log_name="f_loss",
            # ),
            f_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            # f_constraint=lambda: ext.compile_function(
            #     inputs=inputs + extra_inputs,
            #     outputs=constraint_term,
            #     log_name="constraint",
            # ),
            f_loss_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
                log_name="f_loss_constraint",
            ),
        )

    def force_compile(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        # No longer sure how to call these outside of sliced_fun()
        sliced_fun(self._opt_fun["f_loss_constraint"], 1)(inputs, extra_inputs)
        flat_g = sliced_fun(self._opt_fun["f_grad"], 1)(inputs, extra_inputs)
        Hx = self._hvp_approach.build_eval(inputs + extra_inputs)
        Hx(flat_g)

    # def loss(self, inputs, extra_inputs=None):
    #     inputs = tuple(inputs)
    #     if extra_inputs is None:
    #         extra_inputs = tuple()
    #     return sliced_fun(self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)

    # def constraint_val(self, inputs, extra_inputs=None):
    #     inputs = tuple(inputs)
    #     if extra_inputs is None:
    #         extra_inputs = tuple()
    #     return sliced_fun(self._opt_fun["f_constraint"], self._num_slices)(inputs, extra_inputs)

    def loss_constraint(self, inputs, extra_inputs):
        """ Parallelized: same values return in all processes """
        shareds = self.par_objs.shareds
        rank = self.par_data.rank
        avg_fac = self.par_data.avg_fac
        loss, constraint_val = sliced_fun(self._opt_fun["f_loss_constraint"],
            self._num_slices)(inputs, extra_inputs)
        shareds.loss[rank] = avg_fac * loss
        shareds.constraint_val[rank] = avg_fac * constraint_val
        self.par_objs.barriers.loss_cnstr.wait()
        return sum(shareds.loss), sum(shareds.constraint_val)

    def flat_g(self, inputs, extra_inputs):
        """ Parallelized: same values returne din all processes """
        shareds = self.par_objs.shareds
        vb = self.par_data.vb_pair
        shareds.grads_2d[self.par_data.rank, :] = self.par_data.avg_fac * \
            sliced_fun(self._opt_fun["f_grad"], self._num_slices)(inputs, extra_inputs)
        self.par_objs.barriers.flat_g[0].wait()
        shareds.flat_g[vb[0]:vb[1]] = shareds.grads_2d[:, vb[0]:vb[1]].sum(axis=0)
        self.par_objs.barriers.flat_g[1].wait()
        return shareds.flat_g  # (or access this elsewhere)

    # Instead, have the master set this when it assigns paths.
    # def set_avg_fac(self, n_samples):
    #     shareds = self.par_objs.shareds
    #     shareds.n_samples[self.par_data.rank] = n_samples
    #     self.par_objs.barriers.avg_fac.wait()
    #     self.par_data.avg_fac = 1.0 * n_samples / sum(shareds.n_samples)

    def prepare_inputs(self, inputs, extra_inputs, subsample_grouped_inputs):
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

        return inputs, extra_inputs, subsample_inputs

    def optimize(self, inputs, extra_inputs=None, subsample_grouped_inputs=None, avg_fac=None):
        """ Master (rank 0) process executes this method """

        shareds = self.par_objs.shareds

        inputs, extra_inputs, subsample_inputs = self.prepare_inputs(
            inputs, extra_inputs, subsample_grouped_inputs)

        if avg_fac is not None:
            self.par_data.avg_fac = np.array(avg_fac, dtype=self.floatX)

        logger.log("computing loss, mean KL before")
        loss_before, mean_kl_before = self.loss_constraint(inputs, extra_inputs)

        logger.log("performing update")
        logger.log("computing descent direction")
        self.flat_g(inputs, extra_inputs)  # writes to shareds.flat_g
        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)
        krylov.cg(Hx, shareds.flat_g, self.par_objs.cg_par_objs,
            cg_iters=self._cg_iters)
        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (shareds.descent.dot(shareds.flat_g) + 1e-8))
        )
        if not np.isnan(initial_step_size):
            shareds.descent *= initial_step_size
        logger.log("descent direction computed")

        shareds.prev_param[:] = self._target.get_param_values(trainable=True)
        # prev_param = np.copy(self._target.get_param_values(trainable=True))
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            shareds.cur_param[:] = shareds.prev_param - ratio * shareds.descent
            # cur_step = ratio * flat_descent_step
            # cur_param = prev_param - cur_step
            self.par_objs.barriers.bktrk.wait()
            self._target.set_param_values(shareds.cur_param, trainable=True)
            loss, constraint_val = self.loss_constraint(inputs, extra_inputs)
            # loss, constraint_val = sliced_fun(
            #     self._opt_fun["f_loss_constraint"], self._num_slices)(inputs, extra_inputs)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break

        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
                self._max_constraint_val) and not self._accept_violation:
            logger.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                logger.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                logger.log("Violated because constraint %s is NaN" %
                           self._constraint_name)
            if loss >= loss_before:
                logger.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                logger.log(
                    "Violated because constraint %s is violated" % self._constraint_name)
            self._target.set_param_values(shareds.prev_param, trainable=True)

        logger.log("backtrack iters: %d" % n_iter)
        # logger.log("computing loss after")
        logger.log("optimization finished")
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', constraint_val)
        logger.record_tabular('dLoss', loss_before - loss)

    def optimize_worker(self, inputs, extra_inputs=None, subsample_grouped_inputs=None, avg_fac=None):
        """ Workers (rank > 0) execute this synchronously with master """

        shareds = self.par_objs.shareds

        inputs, extra_inputs, subsample_inputs = self.prepare_inputs(
            inputs, extra_inputs, subsample_grouped_inputs)

        if avg_fac is not None:
            self.par_data.avg_fac = np.array(avg_fac, dtype=self.floatX)

        loss_before, mean_kl_before = self.loss_constraint(inputs, extra_inputs)

        self.flat_g(inputs, extra_inputs)  # writes to shareds.flat_g
        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)
        krylov.cg_worker(Hx, self.par_objs.cg_par_objs, cg_iters=self._cg_iters)

        # master write shareds.descent and shareds.prev_param
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            # master write shareds.cur_param
            self.par_objs.barriers.bktrk.wait()
            self._target.set_param_values(shareds.cur_param, trainable=True)
            loss, constraint_val = self.loss_constraint(inputs, extra_inputs)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break

        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
                self._max_constraint_val) and not self._accept_violation:
            self._target.set_param_values(shareds.prev_param, trainable=True)

