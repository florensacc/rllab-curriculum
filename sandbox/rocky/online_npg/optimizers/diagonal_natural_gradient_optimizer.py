from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np
from rllab.policies.base import StochasticPolicy


class DiagonalNaturalGradientOptimizer(Serializable):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=0.1,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param debug_nan: if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected
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

        assert isinstance(target, StochasticPolicy)

        params = target.get_params(trainable=True)
        grads = theano.grad(loss, wrt=params, disconnected_inputs='ignore')
        flat_grad = ext.flatten_tensor_variables(grads)

        obs_var = inputs[0]
        action_var = inputs[1]
        assert obs_var.name == "obs"
        assert action_var.name == "action"
        assert len(target.state_info_keys) == 0

        log_prob = target.distribution.log_likelihood_sym(action_var, target.dist_info_sym(obs_var, dict()))
        sum_log_prob = TT.sum(log_prob)
        grads_log_prob = TT.grad(sum_log_prob, wrt=params)
        flat_grad_log_prob = ext.flatten_tensor_variables(grads_log_prob)

        # diag_fim = (flat_grad_log_prob ** 2) / log_prob.shape[0]

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
            f_log_prob_grad=lambda: ext.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad_log_prob,
                log_name="f_log_prob_grad",
            ),
            # f_diag_fim=lambda: ext.compile_function(
            #     inputs=inputs + extra_inputs,
            #     outputs=diag_fim,
            #     log_name="f_diag_fim",
            # ),
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

    def loss(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def constraint_val(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()
        return self._opt_fun["f_constraint"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None):
        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        logger.log("computing descent direction")

        flat_grad = self._opt_fun["f_grad"](*(inputs + extra_inputs))

        N = inputs[0].shape[0]

        log_prob_grads = []

        for idx in xrange(N):
            sliced = tuple([x[idx:idx+1] for x in inputs]) + extra_inputs
            log_prob_grad = self._opt_fun["f_log_prob_grad"](*sliced)
            log_prob_grads.append(log_prob_grad)
        diag_fim = np.mean(np.square(np.asarray(log_prob_grads)), axis=0)
        # import ipdb; ipdb.set_trace()
        # diag_fim = self._opt_fun["f_diag_fim"](*(inputs + extra_inputs))

        descent_direction = flat_grad / (diag_fim + self._reg_coeff)# + 1.0)#0.1)#self._reg_coeff)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (np.sum(np.square(descent_direction) * diag_fim)))
        )

        # import ipdb; ipdb.set_trace()

        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")
        prev_param = self._target.get_param_values(trainable=True)
        new_param = prev_param - flat_descent_step
        self._target.set_param_values(new_param)
        logger.log("optimization finished")
