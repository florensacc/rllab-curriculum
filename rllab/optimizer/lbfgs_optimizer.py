from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables
from rllab.core.serializable import Serializable
import theano
import scipy.optimize


class LbfgsOptimizer(Serializable):
    """
    Performs unconstrained optimization via L-BFGS.
    """

    def __init__(self, max_opt_itr=20):
        Serializable.quick_init(self, locals())
        self._max_opt_itr = max_opt_itr
        self._opt_fun = None
        self._target = None

    def update_opt(self, loss, target, inputs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        def get_opt_output():
            flat_grad = flatten_tensor_variables(theano.grad(loss, target.get_params(trainable=True)))
            return [loss.astype('float64'), flat_grad.astype('float64')]

        self._opt_fun = lazydict(
            f_loss=lambda: compile_function(inputs, loss),
            f_opt=lambda: compile_function(
                inputs=inputs,
                outputs=get_opt_output(),
            )
        )

    def loss(self, *inputs):
        return self._opt_fun["f_loss"](*inputs)

    def optimize(self, *inputs):
        f_opt = self._opt_fun["f_opt"]

        def f_opt_wrapper(flat_params):
            self._target.set_param_values(flat_params, trainable=True)
            return f_opt(*inputs)

        scipy.optimize.fmin_l_bfgs_b(
            func=f_opt_wrapper, x0=self._target.get_param_values(trainable=True),
            maxiter=self._max_opt_itr
        )
