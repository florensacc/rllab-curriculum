from rllab.misc.ext import compile_function, lazydict, flatten_tensor_variables
from rllab.core.serializable import Serializable
from rllab.algo.first_order_method import parse_update_method
import theano
import time
import numpy as np


class MinibatchOptimizer(Serializable):
    """
    Performs stochastic gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(self, max_epochs=1000, tolerance=1e-6, update_method='sgd', batch_size=32, shuffle_per_epoch=True, callback=None, **kwargs):
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        self._update_method = parse_update_method(update_method, **kwargs)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._shuffle_per_epoch = shuffle_per_epoch

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

        total_size = inputs[0].shape[0]

        itrs_per_epoch = int(np.ceil(total_size * 1.0 / self._batch_size))

        for epoch in xrange(self._max_epochs):
            ids = np.arange(total_size)
            if self._shuffle_per_epoch:
                np.random.shuffle(ids)
            for epoch_itr in xrange(itrs_per_epoch):
                batch_start = epoch_itr * self._batch_size
                batch_end = (epoch_itr + 1) * self._batch_size
                batch = [d[batch_start:batch_end] for d in inputs]
                f_opt(*(batch + extra_inputs))
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
