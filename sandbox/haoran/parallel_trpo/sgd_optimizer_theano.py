from rllab.misc import logger, console, ext
from rllab.core.serializable import Serializable
from collections import OrderedDict
import lasagne
import theano
import theano.tensor as TT
import numpy as np
import time
import pyprind
from functools import partial


class SGDOptimizer(Serializable):
    """
    Same interface as the TBPTT optimizer.
    """

    def __init__(
            self,
            update_method=lasagne.updates.adam,
            learning_rate=1e-3,
            n_epochs=2,
            batch_size=32,
            gradient_clipping=40,
            callback=None,
            verbose=False,
            permute_inputs=True,
            log_prefix="sgd_opt: ",
            **kwargs):
        """

        :param n_epochs:
        :param update_method:
        :param batch_size: None or an integer. If None the whole dataset will be used.
        :param callback:
        :param kwargs:
        :return:
        """
        Serializable.quick_init(self, locals())
        self.opt_fun = None
        self.target = None
        self.callback = callback
        update_method = partial(update_method, learning_rate=learning_rate)
        self._update_method = update_method
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.verbose = verbose
        self.input_vars = None
        self.train_op = None
        self.permute_inputs = permute_inputs
        self.log_prefix = log_prefix

    def log(self, message, color=None):
        if color is not None:
            message = console.colorize(message, color)
        logger.log(self.log_prefix + message)


    def update_opt(self, loss, target, inputs, extra_inputs=None, diagnostic_vars=None, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """

        self.target = target

        if diagnostic_vars is None:
            diagnostic_vars = OrderedDict()

        params = target.get_params(trainable=True)
        gradients = theano.grad(loss, params, disconnected_inputs='ignore')
        if self.gradient_clipping is not None:
            gradients = [
                TT.clip(g, -self.gradient_clipping, self.gradient_clipping)
                for g in gradients
            ]

        updates = self._update_method(gradients, target.get_params(trainable=True))
        updates = OrderedDict([(k, v.astype(k.dtype)) for k, v in updates.items()])

        if extra_inputs is None:
            extra_inputs = list()

        self.input_vars = inputs + extra_inputs
        self.f_train = ext.compile_function(
            inputs=self.input_vars,
            outputs=[loss] + list(diagnostic_vars.values()),
            updates=updates,
        )
        self.f_loss_diagostics = ext.compile_function(
            inputs=self.input_vars,
            outputs=[loss] + list(diagnostic_vars.values()),
        )
        self.diagnostic_vars = diagnostic_vars

    def loss_diagnostics(self, inputs, extra_inputs=None):
        """
        Compute loss and kl one minibatch at a time; then average the results.
        """
        N = len(inputs[0])
        if self.batch_size is None:
            batch_size = N
        else:
            batch_size = self.batch_size
        if extra_inputs is None:
            extra_inputs = list()

        losses = []
        diags = {k: [] for k in self.diagnostic_vars.keys()}

        for batch_idx in range(0, N, batch_size):
            batch_sliced_inputs = [x[batch_idx:batch_idx + batch_size] for x in inputs]
            loss, *diagnostics = self.f_loss_diagostics(*(batch_sliced_inputs + extra_inputs))
            losses.append(loss)
            for k, diag_val in zip(self.diagnostic_vars.keys(), diagnostics):
                diags[k].append(diag_val)
        return np.mean(losses), {k: np.mean(vals) for k, vals in diags.items()}

    def optimize(self, inputs, extra_inputs=None, callback=None):
        """
        For each epoch (also called itr in pposgd_clip_ratio_theano), repeatedly sample a minibatch of inputs and decrease the loss, until all data are used. Meanwhile output diagnostics only using one minibatch. Training stops if self.n_epochs is reached, or callback returns False.
        """

        N = len(inputs[0])
        if self.batch_size is None:
            batch_size = N
        else:
            batch_size = self.batch_size
            if self.permute_inputs:
                new_indices = np.random.permutation(N)
                inputs = [x[new_indices] for x in inputs]
                self.log("permuted inputs")

        if extra_inputs is None:
            extra_inputs = list()


        self.log("Start training...")

        for epoch_id in range(self.n_epochs):
            self.log("Epoch %d" % epoch_id)
            progbar = pyprind.ProgBar(N)
            losses = []
            diags = OrderedDict([(k, []) for k in self.diagnostic_vars.keys()])
            for batch_idx in range(0, N, batch_size):
                batch_sliced_inputs = [x[batch_idx:batch_idx + batch_size] for x in inputs]
                loss, *diagnostics = self.f_train(*(batch_sliced_inputs + extra_inputs))
                losses.append(loss)
                for k, diag_val in zip(self.diagnostic_vars.keys(), diagnostics):
                    diags[k].append(diag_val)
                progbar.update(len(batch_sliced_inputs[0]), force_flush=True)
            if progbar.active:
                progbar.stop()

            diags = OrderedDict([(k, np.mean(vals)) for k, vals in diags.items()])
            loss = np.mean(losses)
            if self.verbose:
                log_message = "Loss: %f" % loss
                for k, v in diags.items():
                    log_message += "; %s: %f" % (k, v)
                self.log(log_message)

            if self.callback or callback:
                callback_args = dict(
                    loss=loss,
                    diagnostics=diags,
                    itr=epoch_id,
                )
                if self.callback:
                    if not self.callback(callback_args):
                        return
                if callback:
                    if not callback(**callback_args):
                        return
