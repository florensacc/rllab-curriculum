from rllab.misc import logger
from rllab.core.serializable import Serializable

import numpy as np
import multiprocessing as mp

from sandbox.adam.parallel.util import SimpleContainer


class ParallelPerlmutterHvp(Serializable):
    """
    Only difference from serial is the function defined within build_eval().
    """

    def __init__(self, num_slices=1):
        Serializable.quick_init(self, locals())
        self.target = None
        self.reg_coeff = None
        self.opt_fun = None
        self._num_slices = num_slices
        self.pd = None  # relies on ParallelCGOpt class to set parallel values.
        self._par_objs = None

    def update_opt(self, f, target, inputs, reg_coeff):
        import theano.tensor as TT
        import theano
        from rllab.misc import ext
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
        from rllab.misc.ext import sliced_fun
        def parallel_eval(x):
            """
            Parallelized.
            """
            shareds, barriers = self._par_objs

            xs = tuple(self.target.flat_to_params(x, trainable=True))

            shareds.grads_2d[:, self.pd.rank] = self.pd.avg_fac * \
                                                sliced_fun(self.opt_fun["f_Hx_plain"], self._num_slices)(inputs,
                                                                                                         xs)
            barriers.Hx[0].wait()

            shareds.Hx[self.pd.vb[0]:self.pd.vb[1]] = \
                self.reg_coeff * x[self.pd.vb[0]:self.pd.vb[1]] + \
                np.sum(shareds.grads_2d[self.pd.vb[0]:self.pd.vb[1], :], axis=1)
            barriers.Hx[1].wait()
            return shareds.Hx  # (or can just access this persistent var elsewhere)

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

    Only the optimize() method changes for parallel implementation, but some
    options of serial implementation may not be available.
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
            num_slices=1,
            name=None,
    ):
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
        # if hvp_approach is None:
        #     hvp_approach = PerlmutterHvp(num_slices)
        hvp_approach = ParallelPerlmutterHvp(num_slices)  # Only option supported.
        self._hvp_approach = hvp_approach

        self._name = name

    def update_opt(self, loss, target, leq_constraint, inputs, extra_inputs=None, constraint_name="constraint",
                   *args,
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
        import theano
        from rllab.misc import ext
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
            f_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
            ),
            f_loss_constraint=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=[loss, constraint_term],
                log_name="f_loss_constraint",
            ),
        )

    def init_par_objs(self, n_parallel, size_grad):
        """
        These objects will be inherited by forked subprocesses.
        (Also possible to return these and attach them explicitly within
        subprocess--needed in Windows.)
        """
        n_grad_elm_worker = -(-size_grad // n_parallel)  # ceiling div
        vb_idx = [n_grad_elm_worker * i for i in range(n_parallel + 1)]
        vb_idx[-1] = size_grad

        # Build container to share with hvp_approach.
        par_data = SimpleContainer(
            rank=None,
            avg_fac=1.0 / n_parallel,
            vb=[(vb_idx[i], vb_idx[i + 1]) for i in range(n_parallel)],
        )
        self.pd = par_data
        self._hvp_approach.pd = par_data

        shareds = SimpleContainer(
            flat_g=np.frombuffer(mp.RawArray('d', size_grad)),
            grads_2d=np.reshape(
                np.frombuffer(mp.RawArray('d', size_grad * n_parallel)),
                (size_grad, n_parallel)),
            Hx=np.frombuffer(mp.RawArray('d', size_grad)),
            loss=mp.RawArray('d', n_parallel),
            constraint_val=mp.RawArray('d', n_parallel),
            descent=np.frombuffer(mp.RawArray('d', size_grad)),
            prev_param=np.frombuffer(mp.RawArray('d', size_grad)),
            cur_param=np.frombuffer(mp.RawArray('d', size_grad)),
            cg_p=np.frombuffer(mp.RawArray('d', size_grad)),
            n_steps_collected=mp.RawArray('i', n_parallel),
        )
        barriers = SimpleContainer(
            avg_fac=mp.Barrier(n_parallel),
            flat_g=[mp.Barrier(n_parallel) for _ in range(2)],
            loss=mp.Barrier(n_parallel),
            cnstr=mp.Barrier(n_parallel),
            loss_cnstr=mp.Barrier(n_parallel),
            bktrk=mp.Barrier(n_parallel),
        )
        self._par_objs = (shareds, barriers)

        shareds_hvp = SimpleContainer(
            grads_2d=shareds.grads_2d,  # OK to use the same memory.
            Hx=shareds.Hx,
        )
        barriers_hvp = SimpleContainer(
            Hx=[mp.Barrier(n_parallel) for _ in range(2)],
        )
        self._hvp_approach._par_objs = (shareds_hvp, barriers_hvp)

        # For passing into krylov.cg
        cg_par_objs = {
            'z': shareds.Hx,  # (location of result of Hx)
            'p': np.frombuffer(mp.RawArray('d', size_grad)),
            'x': shareds.descent,  # (location to write krylov.cg result)
            'brk': mp.RawValue('i'),
            'barrier': mp.Barrier(n_parallel),
        }
        self._cg_par_objs = cg_par_objs

    def init_rank(self, rank):
        self.pd.rank = rank
        self.rank = rank
        self.pd.vb = self.pd.vb[rank]  # (assign gradient vector boundaries)
        self.vb = self.pd.vb

    def set_avg_fac(self, n_steps_collected):
        shareds, barriers = self._par_objs
        shareds.n_steps_collected[self.rank] = n_steps_collected
        barriers.avg_fac.wait()
        self.pd.avg_fac = 1.0 * n_steps_collected / sum(shareds.n_steps_collected)
        self.avg_fac = self.pd.avg_fac

    def _loss(self, inputs, extra_inputs):
        """
        Parallelized: returns the same value in all workers.
        """
        from rllab.misc.ext import sliced_fun
        shareds, barriers = self._par_objs
        shareds.loss[self.rank] = self.avg_fac * sliced_fun(
            self._opt_fun["f_loss"], self._num_slices)(inputs, extra_inputs)
        barriers.loss.wait()
        return sum(shareds.loss)

    def _constraint_val(self, inputs, extra_inputs):
        """
        Parallelized: returns the same value in all workers.
        """
        from rllab.misc.ext import sliced_fun
        shareds, barriers = self._par_objs
        shareds.constraint_val[self.rank] = self.avg_fac * sliced_fun(
            self._opt_fun["f_constraint"], self._num_slices)(inputs, extra_inputs)
        barriers.cnstr.wait()
        return sum(shareds.constraint_val)

    def _loss_constraint(self, inputs, extra_inputs):
        """
        Parallelized: returns the same values in all workers.
        """
        from rllab.misc.ext import sliced_fun

        shareds, barriers = self._par_objs
        loss, constraint_val = sliced_fun(self._opt_fun["f_loss_constraint"],
                                          self._num_slices)(inputs, extra_inputs)
        shareds.loss[self.rank] = self.avg_fac * loss
        shareds.constraint_val[self.rank] = self.avg_fac * constraint_val
        barriers.loss_cnstr.wait()
        return sum(shareds.loss), sum(shareds.constraint_val)

    def _flat_g(self, inputs, extra_inputs):
        """
        Parallelized: returns the same values in all workers.
        """
        from rllab.misc.ext import sliced_fun
        shareds, barriers = self._par_objs
        # Each worker records result available to all.
        shareds.grads_2d[:, self.rank] = self.avg_fac * \
                                         sliced_fun(self._opt_fun["f_grad"], self._num_slices)(inputs,
                                                                                               extra_inputs)
        barriers.flat_g[0].wait()
        # Each worker sums over an equal share of the grad elements across
        # workers (row major storage--sum along rows).
        shareds.flat_g[self.vb[0]:self.vb[1]] = \
            np.sum(shareds.grads_2d[self.vb[0]:self.vb[1], :], axis=1)
        barriers.flat_g[1].wait()
        # No return (elsewhere, access shareds.flat_g)

    def force_compile(self, inputs, extra_inputs=None):
        """
        Serial - force compiling of Theano functions.
        """
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        self._opt_fun["f_loss"](*(inputs + extra_inputs))
        x = self._opt_fun["f_grad"](*(inputs + extra_inputs))
        xs = tuple(self._target.flat_to_params(x, trainable=True))
        self._hvp_approach.opt_fun["f_Hx_plain"](*(inputs + extra_inputs + xs))
        self._opt_fun["f_loss_constraint"](*(inputs + extra_inputs))
        # self._opt_fun["f_constraint"]  # not used!

    def optimize(self, inputs, extra_inputs=None, subsample_grouped_inputs=None):
        """
        Parallelized: all workers get the same parameter update.
        """
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

        if self.rank == 0:
            self._optimize_master(inputs, extra_inputs, subsample_inputs)
        else:
            self._optimize(inputs, extra_inputs, subsample_inputs)

    def log(self, message):
        if self._name is not None:
            logger.log(self._name + ": " + message)
        else:
            logger.log(message)

    def _optimize_master(self, inputs, extra_inputs, subsample_inputs):
        from sandbox.adam.parallel import krylov
        shareds, barriers = self._par_objs

        self.log("computing loss before")
        loss_before = self._loss(inputs, extra_inputs)  # (parallel)
        self.log("performing update")
        self.log("computing descent direction")
        self._flat_g(inputs, extra_inputs)  # (parallel, writes shareds.flat_g)
        # Hx is parallelized.
        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)
        # Krylov is parallelized, but only to save on memory (could run serial).
        krylov.cg(Hx, shareds.flat_g, self._cg_par_objs, self.rank,
                  cg_iters=self._cg_iters)
        # # Serial call version:
        # shareds.descent[:] = krylov.cg(Hx, shareds.flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val *
            (1. / (shareds.descent.dot(shareds.flat_g) + 1e-8))
        )
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        shareds.descent *= initial_step_size
        self.log("descent direction computed")
        shareds.prev_param[:] = self._target.get_param_values(trainable=True)

        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            shareds.cur_param[:] = shareds.prev_param - ratio * shareds.descent
            barriers.bktrk.wait()
            self._target.set_param_values(shareds.cur_param, trainable=True)
            loss, constraint_val = self._loss_constraint(inputs, extra_inputs)  # (parallel)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break

        if ((np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or
                     constraint_val >= self._max_constraint_val) and not self._accept_violation):
            self._target.set_param_values(shareds.prev_param, trainable=True)
            self.log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                self.log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                self.log("Violated because constraint %s is NaN" %
                         self._constraint_name)
            if loss >= loss_before:
                self.log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                self.log(
                    "Violated because constraint %s is violated" % self._constraint_name)
            loss = loss_before
            constraint_val = 0.

        self.log("backtrack iters: %d" % n_iter)
        self.log("optimization finished")

        if self._name is not None:
            log_prefix = self._name + "_"
        else:
            log_prefix = ""
        logger.record_tabular(log_prefix + 'LossBefore', loss_before)
        logger.record_tabular(log_prefix + 'LossAfter', loss)
        # logger.record_tabular(log_prefix + 'MeanKLBefore', mean_kl_before)  # zero!
        logger.record_tabular(log_prefix + 'MeanKL', constraint_val)
        logger.record_tabular(log_prefix + 'dLoss', loss_before - loss)
        logger.record_tabular(log_prefix + 'bktrk_iters', n_iter)

    def _optimize(self, inputs, extra_inputs, subsample_inputs):
        from sandbox.adam.parallel import krylov

        shareds, barriers = self._par_objs

        loss_before = self._loss(inputs, extra_inputs)  # (parallel)
        self._flat_g(inputs, extra_inputs)  # (parallel, writes shareds.flat_g)
        Hx = self._hvp_approach.build_eval(subsample_inputs + extra_inputs)
        # Krylov is parallelized, but only to save on memory (could run serial).
        krylov.cg(Hx, shareds.flat_g, self._cg_par_objs, self.rank,
                  cg_iters=self._cg_iters)
        # Serial call version:
        # _ = krylov.cg(Hx, shareds.flat_g, cg_iters=self._cg_iters)

        # master writes shareds.descent and shareds.prev_param

        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            # master writes shareds.cur_param
            barriers.bktrk.wait()
            self._target.set_param_values(shareds.cur_param, trainable=True)
            loss, constraint_val = self._loss_constraint(inputs, extra_inputs)  # (parallel)
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break

        if ((np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or
                     constraint_val >= self._max_constraint_val) and not self._accept_violation):
            self._target.set_param_values(shareds.prev_param, trainable=True)
