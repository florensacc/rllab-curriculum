from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import numpy as np
import scipy.optimize


class PenaltyLbfgs(Serializable):
    """
    Performs constrained optimization via penalized L-BFGS. The penalty term is adaptively adjusted to make sure that
    the constraint is satisfied.
    """

    def __init__(
            self,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            max_penalty_itr=10,
            adapt_penalty=True):
        Serializable.quick_init(self, locals())
        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._max_opt_itr = max_opt_itr
        self._max_penalty_itr = max_penalty_itr
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None

    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name="constraint"):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        constraint_term, constraint_value = leq_constraint
        penalty_var = TT.scalar("penalty")
        penalized_loss = loss + penalty_var * constraint_term

        self._target = target
        self._max_constraint_val = constraint_value
        self._constraint_name = constraint_name

        def get_opt_output():
            flat_grad = flatten_tensor_variables(theano.grad(loss, target.get_params(trainable=True)))
            return [loss.astype('float'), flat_grad.astype('float')]

        self._opt_fun = lazydict(
            f_loss=lambda: compile_function(inputs, loss),
            f_constraint=lambda: compile_function(inputs, constraint_term),
            f_penalized_loss=lambda: compile_function(
                inputs=inputs + [penalty_var],
                outputs=[loss, constraint_term, penalized_loss]
            ),
            f_opt=lambda: compile_function(
                inputs=inputs + [penalty_var],
                outputs=get_opt_output(),
            )
        )

    def loss(self, *inputs):
        return self._opt_fun["f_loss"](*inputs)

    def optimize(self, *inputs):
        try_penalty = np.clip(
            self._penalty, self._min_penalty, self._max_penalty)

        penalty_scale_factor = None
        opt_params = None
        f_opt = self._opt_fun["f_opt"]
        f_penalized_loss = self._opt_fun["f_penalized_loss"]

        def gen_f_opt(penalty):
            def f(flat_params):
                self._target.set_param_values(flat_params)
                return f_opt(*(inputs + (penalty,)))
            return f

        cur_params = self._target.get_param_values(trainable=True)

        for penalty_itr in range(self._max_penalty_itr):
            logger.log('trying penalty=%.3f...' % try_penalty)

            itr_opt_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
                func=gen_f_opt(try_penalty), x0=cur_params,
                maxiter=self._max_opt_itr
            )

            _, try_loss, try_constraint_val = f_penalized_loss(*(inputs + (try_penalty,)))

            logger.log('penalty %f => loss %f, %s %f' %
                       (try_penalty, try_loss, self._constraint_name, try_constraint_val))

            if try_constraint_val < self._max_constraint_val or \
                    (penalty_itr == self._max_penalty_itr - 1 and opt_params is None):
                opt_params = itr_opt_params
                self._penalty = try_penalty

            if not self._adapt_penalty:
                break

            # decide scale factor on the first iteration
            if penalty_scale_factor is None or np.isnan(try_constraint_val):
                if try_constraint_val > self._max_constraint_val or np.isnan(try_constraint_val):
                    # need to increase penalty
                    penalty_scale_factor = self._increase_penalty_factor
                else:
                    # can shrink penalty
                    penalty_scale_factor = self._decrease_penalty_factor
            else:
                if penalty_scale_factor > 1 and \
                        try_constraint_val <= self._max_constraint_val:
                    break
                elif penalty_scale_factor < 1 and \
                        try_constraint_val >= self._max_constraint_val:
                    break
            try_penalty *= penalty_scale_factor
            if try_penalty < self._min_penalty or \
                    try_penalty > self._max_penalty:
                try_penalty = np.clip(
                    try_penalty, self._min_penalty, self._max_penalty)
                self._penalty = try_penalty
