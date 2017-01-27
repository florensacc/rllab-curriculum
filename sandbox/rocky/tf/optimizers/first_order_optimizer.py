


from rllab.misc import ext
from rllab.misc import logger
from rllab.core.serializable import Serializable
from rllab.misc.ext import sliced_fun
from sandbox.rocky.tf.misc import tensor_utils
# from rllab.algo.first_order_method import parse_update_method
from rllab.optimizers.minibatch_dataset import BatchDataset
from collections import OrderedDict
import tensorflow as tf
import time
from functools import partial
import pyprind


class FirstOrderOptimizer(Serializable):
    """
    Performs (stochastic) gradient descent, possibly using fancier methods like adam etc.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            # learning_rate=1e-3,
            max_epochs=1000,
            max_updates=None,
            tolerance=1e-6,
            batch_size=32,
            callback=None,
            verbose=False,
            n_slices=1,
            n_passes_per_epoch=1,
            **kwargs):
        """

        :param max_epochs:
        :param tolerance:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self._opt_fun = None
        self._target = None
        self._callback = callback
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._max_updates = max_updates
        self._tolerance = tolerance
        self._batch_size = batch_size
        self._verbose = verbose
        self._input_vars = None
        self._train_op = None
        self._n_slices = n_slices
        self._n_passes_per_epoch = n_passes_per_epoch

    def update_opt(self, loss, target, inputs, extra_inputs=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self._target = target

        self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params(trainable=True))

        # updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.iteritems()])

        if extra_inputs is None:
            extra_inputs = list()
        self._input_vars = inputs + extra_inputs
        self._opt_fun = ext.lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs + extra_inputs, loss),
        )

    def loss(self, inputs, extra_inputs=None):
        if extra_inputs is None:
            extra_inputs = tuple()
        return sliced_fun(self._opt_fun["f_loss"], self._n_slices)(inputs, extra_inputs)

    def optimize(self, inputs, extra_inputs=None, callback=None):

        if len(inputs) == 0:
            # Assumes that we should always sample mini-batches
            raise NotImplementedError

        if extra_inputs is None:
            extra_inputs = tuple()

        if self._verbose:
            logger.log("Computing loss before")
        last_loss = self.loss(inputs, extra_inputs)
        if self._verbose:
            logger.log("Computed loss before")

        start_time = time.time()

        dataset = BatchDataset(inputs, self._batch_size, extra_inputs=extra_inputs)

        sess = tf.get_default_session()

        n_updates = 0

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))
                progbar = pyprind.ProgBar(len(inputs[0]) * self._n_passes_per_epoch)

            for _ in range(self._n_passes_per_epoch):
                for batch in dataset.iterate(update=True):
                    sess.run(self._train_op, dict(list(zip(self._input_vars, batch))))
                    n_updates += 1
                    if self._verbose:
                        progbar.update(len(batch[0]))
                    if self._max_updates is not None and n_updates >= self._max_updates:
                        break

            if self._verbose:
                if progbar.active:
                    progbar.stop()

            new_loss = self.loss(inputs, extra_inputs)

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            if self._callback or callback:
                elapsed = time.time() - start_time
                callback_args = dict(
                    loss=new_loss,
                    params=self._target.get_param_values(trainable=True) if self._target else None,
                    itr=epoch,
                    elapsed=elapsed,
                )
                if self._callback:
                    self._callback(callback_args)
                if callback:
                    callback(**callback_args)

            if self._tolerance is not None and abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

            if self._max_updates is not None and n_updates >= self._max_updates:
                break
