from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables
from rllab.core.serializable import Serializable
from rllab.algo.first_order_method import parse_update_method
from rllab.optimizer.minibatch_dataset import MinibatchDataset
import time


class MinibatchOptimizer(Serializable):
    """
    Performs stochastic gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(self, max_epochs=1000, tolerance=1e-6, update_method='sgd', batch_size=32, callback=None, **kwargs):
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        self._update_method = parse_update_method(update_method, **kwargs)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size

    def update_opt(self, loss, target, inputs, extra_inputs=None):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        updates = self._update_method(loss, target.get_params(trainable=True))

        if extra_inputs is None:
            extra_inputs = list()

        self._opt_fun = lazydict(
            f_loss=lambda: compile_function(inputs + extra_inputs, loss),
            f_opt=lambda: compile_function(
                inputs=inputs + extra_inputs,
                outputs=loss,
                updates=updates,
            )
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = list()
        return self._opt_fun["f_loss"](*(inputs + extra_inputs))

    def optimize(self, inputs, extra_inputs=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        f_opt = self._opt_fun["f_opt"]
        f_loss = self._opt_fun["f_loss"]

        if extra_inputs is None:
            extra_inputs = list()

        last_loss = f_loss(*(inputs + extra_inputs))

        start_time = time.time()

        dataset = MinibatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)

        for epoch in xrange(self._max_epochs):
            for batch in dataset.iterate(update=True):
                f_opt(*batch)

            new_loss = f_loss(*(inputs + extra_inputs))

            if self._callback:
                elapsed = time.time() - start_time
                self._callback(dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True),
                    itr=epoch,
                    elapsed=elapsed,
                ))

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss
