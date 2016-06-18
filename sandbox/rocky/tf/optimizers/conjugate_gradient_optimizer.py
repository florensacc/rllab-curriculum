from __future__ import print_function
from __future__ import absolute_import
from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
from rllab.core.serializable import Serializable
import tensorflow as tf
import itertools
import numpy as np


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
            subsample_factor=0.1,
            backtrack_ratio=0.8,
            max_backtracks=15):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
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
        grads = tf.gradients(loss, params)
        flat_grad = tensor_utils.flatten_tensor_variables(grads)

        constraint_grads = tf.gradients(constraint_term, params)
        xs = tuple([tensor_utils.new_tensor_like("random", p) for p in params])

        Hx_plain_splits = tf.gradients(
            tf.reduce_sum(
                tf.pack([tf.reduce_sum(g * x) for g, x in itertools.izip(constraint_grads, xs)])
            ),
            params
        )
        Hx_plain = tensor_utils.flatten_tensor_variables(Hx_plain_splits)

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                log_name="f_loss",
            ),
            f_grad=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=flat_grad,
                log_name="f_grad",
            ),
            f_Hx_plain=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs + xs,
                outputs=Hx_plain,
                log_name="f_Hx_plain",
            ),
            f_constraint=lambda: tensor_utils.compile_function(
                inputs=inputs + extra_inputs,
                outputs=constraint_term,
                log_name="constraint",
            ),
            f_loss_constraint=lambda: tensor_utils.compile_function(
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

        if self._subsample_factor < 1:
            n_samples = len(inputs[0])
            inds = np.random.choice(
                n_samples, n_samples * self._subsample_factor, replace=False)
            subsample_inputs = tuple([x[inds] for x in inputs])
        else:
            subsample_inputs = inputs

        logger.log("computing loss before")
        loss_before = self._opt_fun["f_loss"](*(inputs + extra_inputs))
        logger.log("performing update")
        logger.log("computing descent direction")

        flat_g = self._opt_fun["f_grad"](*(inputs + extra_inputs))

        def Hx(x):
            with logger.log_time("Hx"):
                xs = tuple(self._target.flat_to_params(x, trainable=True))
                #     rop = f_Hx_rop(*(inputs + xs))
                plain = self._opt_fun["f_Hx_plain"](*(subsample_inputs + extra_inputs + xs)) + self._reg_coeff * x
                # assert np.allclose(rop, plain)
                return plain
                # alternatively we can do finite difference on flat_grad

        descent_direction = krylov.cg(Hx, flat_g, cg_iters=self._cg_iters)

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        flat_descent_step = initial_step_size * descent_direction

        logger.log("descent direction computed")

        prev_param = self._target.get_param_values(trainable=True)
        n_iter = 0
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = self._opt_fun["f_loss_constraint"](*(inputs + extra_inputs))
            if loss < loss_before and constraint_val <= self._max_constraint_val:
                break
        logger.log("backtrack iters: %d" % n_iter)
        logger.log("computing loss after")
        logger.log("optimization finished")
