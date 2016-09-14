from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
# from rllab.misc.ext import flatten_tensor_variables
import itertools
import numpy as np
import tensorflow as tf
from sandbox.rocky.tf.misc import tensor_utils

from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp, PerlmutterHvp
from rllab.sampler.parallel_sampler import _get_scoped_G
from rllab.sampler.stateful_pool import singleton_pool
import pickle

"""
Be careful not to clone inputs to each worker. That takes too much memory.
"""

def log(message):
    logger.log("CG: " + message, with_prefix=False)

class ConjugateGradientOptimizerMP(Serializable):
    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            accept_violation=False,
            hvp_approach=None):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param debug_nan: if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected
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

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan
        self._accept_violation = accept_violation
        if hvp_approach is None:
            hvp_approach = PerlmutterHvp()
        self._hvp_approach = hvp_approach

    def worker_copy(self):
        """
        Return a non-parallel copy of itself.
        Make sure that these two classes have the same interface.
        """
        state = self.__getstate__()
        copy = ConjugateGradientOptimizer(*state["__args"],**state["__kwargs"])
        return copy


    def f(self, f_name):
        """
        f_name: "f_loss", "f_grad", "f_constraint", "f_loss_constraint"
        """

        total_sample_count = np.sum(self.n_worker_samples)
        collected = singleton_pool.run_collect(
            _worker_collect_f,
            threshold=total_sample_count,
            args=(f_name, None),
            show_prog_bar=True,
        )

        # Assume "value" is a tuple of (numbers of np arrays)
        # Do not use zip(results_list, self.n_worker_samples) here, because potentially the results are not ordered in the way inputs are distributed
        if isinstance(collected[0][0],float) or isinstance(collected[0][0],np.ndarray):
            v = None
            for result,sample_count in collected:
                w = result * sample_count / float(total_sample_count)
                if v is None:
                    v = w
                else:
                    v += w
            return v
        else:
            combined_results = None
            for results, sample_count in collected:
                if combined_results is None:
                    combined_results = [
                        v * sample_count/float(total_sample_count)
                        for v in results
                    ]
                else:
                    combined_results = [
                        v0 + v * sample_count / float(total_sample_count)
                        for v0,v in zip(combined_results,results)
                    ]
            return tuple(combined_results)

    def distribute_inputs(self, sliced_inputs, nonsliced_inputs,
        subsample_sliced_inputs, subsample_nonsliced_inputs):
        """
        Cut inputs into slices and feed them to parallel workers
        See rllab.misc.ext.sliced_fun for inspiration
        Todo: delete a slice from the main process once it gets distributed
        """

        if nonsliced_inputs is None:
            nonsliced_inputs = []
        if isinstance(nonsliced_inputs, tuple):
            nonsliced_inputs = list(nonsliced_inputs)
        n_samples = len(sliced_inputs[0])
        n_subsamples = len(subsample_sliced_inputs[0])
        n_slices = singleton_pool.n_parallel
        slice_size = max(1, n_samples // n_slices)
        subsample_slice_size = max(1, n_subsamples // n_slices)
        self.n_worker_samples = []
        self.n_worker_subsamples = []
        args = []

        for start, subsample_start in zip(
            range(0, n_samples // slice_size * slice_size, slice_size),
            range(0, n_subsamples // subsample_slice_size * subsample_slice_size, subsample_slice_size)
        ):
            inputs_slice = [v[start:start + slice_size] for v in sliced_inputs]
            self.n_worker_samples.append(len(inputs_slice[0]))

            if subsample_sliced_inputs is not None:
                subsample_inputs_slice = [
                    v[subsample_start : subsample_start + subsample_slice_size]
                    for v in subsample_sliced_inputs
                ]
                self.n_worker_subsamples.append(len(subsample_inputs_slice[0]))
            else:
                subsample_inputs_slice = None
                self.n_worker_subsamples.append(len([input_slice[0]]))
            args.append([inputs_slice, nonsliced_inputs, subsample_inputs_slice, subsample_nonsliced_inputs, None])
        singleton_pool.run_each(
            _worker_set_inputs,
            args
        )


    def loss(self):
        return self.f("f_loss")

    def constraint_val(self):
        return self.f("f_constraint")

    def loss_constraint_val(self):
        return self.f("f_loss_constraint")

    def prepare_optimization(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        # subsample for Hx
        self.subsample_for_hvp = self._subsample_factor < 1 - 1e-5
        if self.subsample_for_hvp:
            subsample_grouped_inputs = [inputs]
            subsample_inputs = tuple()
            for inputs_grouped in subsample_grouped_inputs:
                n_samples = len(inputs_grouped[0])
                inds = np.random.choice(
                    n_samples, int(n_samples * self._subsample_factor), replace=False)
                subsample_inputs += tuple([x[inds] for x in inputs_grouped])
            subsample_extra_inputs = extra_inputs # need deepcopy?
        else:
            subsample_inputs = None
            subsample_extra_inputs = None

        # distribute inputs
        log("Distributing inputs to workers.")
        self.distribute_inputs(
            inputs, extra_inputs,
            subsample_inputs, subsample_extra_inputs
        )
        log("Distributed")


    def optimize(self,loss_before=None):
        prev_param = np.copy(self._target.get_param_values(trainable=True))
        set_policy_trainable_params(prev_param)
        log("Start CG optimization: #parameters: %d, #inputs: %d, #subsample_inputs: %d"%(len(prev_param),np.sum(self.n_worker_samples), np.sum(self.n_worker_subsamples)))

        log("computing gradient")
        flat_g = self.f("f_grad")
        log("gradient computed")

        log("Building Hessian-vector-product for workers")
        singleton_pool.run_each(
            _worker_build_hvp,
            [(self.subsample_for_hvp,None)] * singleton_pool.n_parallel
        )
        def Hx(x):
            values = singleton_pool.run_collect(
                _worker_collect_Hx,
                threshold=singleton_pool.n_parallel,
                args=(x,None),
                show_prog_bar=True,
            )
            v = None
            if self.subsample_for_hvp:
                sample_counts = self.n_worker_samples
            else:
                sample_counts = self.n_worke
            for value, sample_count in zip(values,self.n_worker_samples):
                if v is None:
                    v = value * sample_count
                else:
                    v += value * sample_count
            v = v * (1./np.sum(self.n_worker_samples))
            return v
        log("Done")


        log("computing descent direction by CG")
        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)
        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        ) # can avoid computing Hx again?
        if np.isnan(initial_step_size):
            initial_step_size = 1.
        flat_descent_step = initial_step_size * descent_direction
        log("descent direction computed")


        if loss_before is None:
            loss_before = self.f("f_loss")
        log("start backtracking to fit the constraint")
        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            set_policy_trainable_params(cur_param)
            final_param = cur_param
            loss, constraint_val = self.f("f_loss_constraint")
            if self._debug_nan and np.isnan(constraint_val):
                import ipdb;
                ipdb.set_trace()
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
            self._max_constraint_val) and not self._accept_violation:
            log("Line search condition violated. Rejecting the step!")
            if np.isnan(loss):
                log("Violated because loss is NaN")
            if np.isnan(constraint_val):
                log("Violated because constraint %s is NaN" % self._constraint_name)
            if loss >= loss_before:
                log("Violated because loss not improving")
            if constraint_val >= self._max_constraint_val:
                log("Violated because constraint %s is violated" % self._constraint_name)
            set_policy_trainable_params(prev_param)
            final_param = prev_param
        log("backtrack iters: %d" % n_iter)
        self._target.set_param_values(final_param)
        log("optimization finished")


def _worker_set_policy_trainable_params(G,params,scope=None):
    G = _get_scoped_G(G, scope)
    G.policy.set_param_values(params,trainable=True)

def set_policy_trainable_params(params, scope=None):
    singleton_pool.run_each(
        _worker_set_policy_trainable_params,
        [(params,scope)] * singleton_pool.n_parallel
    )

def _worker_set_inputs(G, sliced_inputs, nonsliced_inputs,
    subsample_sliced_inputs, subsample_nonsliced_inputs, scope=None):
    G = _get_scoped_G(G, scope)
    G.sliced_inputs = sliced_inputs
    G.nonsliced_inputs = nonsliced_inputs
    G.subsample_sliced_inputs = subsample_sliced_inputs
    G.subsample_nonsliced_inputs = subsample_nonsliced_inputs
    G.n_samples = len(sliced_inputs[0])
    G.n_subsamples = len(subsample_sliced_inputs[0])
    log("A worker gets %d samples and %d subsamples"%(G.n_samples, G.n_subsamples))

def _worker_collect_f(G, f_name, scope=None):
    G = _get_scoped_G(G, scope)
    results = G.optimizer._opt_fun[f_name](*(G.sliced_inputs + G.nonsliced_inputs))
    sample_count = len(G.sliced_inputs[0])
    return (results, sample_count), sample_count

def _worker_build_hvp(G,subsample_for_hvp,scope):
    G = _get_scoped_G(G,scope)
    if subsample_for_hvp:
        Hx = G.optimizer._hvp_approach.build_eval(
            tuple(G.subsample_sliced_inputs) + tuple(G.subsample_nonsliced_inputs)
        )
    else:
        Hx = G.optimizer._hvp_approach.build_eval(
            tuple(G.sliced_inputs) + tuple(G.nonsliced_inputs)
        )
    G.optimizer._opt_fun["Hx"] = lambda: Hx

def _worker_collect_Hx(G,x,scope):
    G = _get_scoped_G(G, scope)
    value = G.optimizer._opt_fun["Hx"](x)
    return value,1
