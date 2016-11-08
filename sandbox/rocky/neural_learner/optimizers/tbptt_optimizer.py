from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
# from rllab.algo.first_order_method import parse_update_method
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import time
import pyprind


class TBPTTOptimizer(Serializable):
    """
    Performs truncated back-propagation through time.
    """

    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            n_epochs=2,
            batch_size=32,
            n_steps=20,
            gradient_clipping=40,
            callback=None,
            verbose=False,
            learning_rate=1e-3,
            anneal_learning_rate=False,
            no_improvement_tolerance=10,
            anneal_ratio=0.5,
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
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()  # learning_rate=learning_rate)
        self.tf_optimizer_cls = tf_optimizer_cls
        self.tf_optimizer_args = tf_optimizer_args
        self.tf_optimizer = None  # tf_optimizer_cls(**tf_optimizer_args)
        self.learning_rate = learning_rate
        self.anneal_learning_rate = anneal_learning_rate
        self.no_improvement_tolerance = no_improvement_tolerance
        self.anneal_ratio = anneal_ratio
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.gradient_clipping = gradient_clipping
        self.verbose = verbose
        self.input_vars = None
        self.train_op = None

    def update_opt(self, loss, target, inputs, rnn_state_input, rnn_init_state, rnn_final_state, extra_inputs=None,
                   diagnostic_vars=None, **kwargs):
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

        lr_var = tf.placeholder(dtype=tf.float32, shape=(), name="lr")

        self.tf_optimizer = self.tf_optimizer_cls(learning_rate=lr_var, **self.tf_optimizer_args)

        params = target.get_params(trainable=True)

        gvs = self.tf_optimizer.compute_gradients(loss, var_list=params)

        if self.gradient_clipping is not None:
            capped_gvs = [
                (tf.clip_by_value(grad, -self.gradient_clipping, self.gradient_clipping), var)
                if grad is not None else (grad, var)
                for grad, var in gvs
                ]
        else:
            capped_gvs = gvs

        train_op = self.tf_optimizer.apply_gradients(capped_gvs)

        if extra_inputs is None:
            extra_inputs = list()

        self.input_vars = inputs + extra_inputs
        self.rnn_init_state = rnn_init_state
        self.lr_var = lr_var

        self.f_train = tensor_utils.compile_function(
            inputs=self.input_vars + [rnn_state_input, lr_var],
            outputs=[train_op, loss, rnn_final_state] + list(diagnostic_vars.values()),
        )
        self.f_loss_diagnostics = tensor_utils.compile_function(
            inputs=self.input_vars + [rnn_state_input],
            outputs=[loss, rnn_final_state] + list(diagnostic_vars.values()),
        )
        self.diagnostic_vars = diagnostic_vars

    def loss_diagnostics(self, inputs, extra_inputs=None, prev_states=None):
        N, T, _ = inputs[0].shape
        if self.n_steps is None:
            n_steps = T
        else:
            n_steps = self.n_steps
        if self.batch_size is None:
            batch_size = N
        else:
            batch_size = self.batch_size
        if extra_inputs is None:
            extra_inputs = list()

        losses = []
        diags = {k: [] for k in self.diagnostic_vars.keys()}

        for batch_idx in range(0, N, batch_size):
            batch_sliced_inputs = [x[batch_idx:batch_idx + self.batch_size] for x in inputs]
            if prev_states is None:
                states = np.tile(
                    self.rnn_init_state.eval().reshape((1, -1)),
                    (batch_sliced_inputs[0].shape[0], 1)
                )
            else:
                states = prev_states[batch_idx:batch_idx + self.batch_size]
            for t in range(0, T, n_steps):
                time_sliced_inputs = [x[:, t:t + n_steps] for x in batch_sliced_inputs]
                loss, states, *diagnostics = self.f_loss_diagnostics(*(time_sliced_inputs + extra_inputs + [
                    states]))
                losses.append(loss)
                for k, diag_val in zip(self.diagnostic_vars.keys(), diagnostics):
                    diags[k].append(diag_val)
        return np.mean(losses), {k: np.mean(vals) for k, vals in diags.items()}

    def optimize(self, inputs, extra_inputs=None, callback=None, prev_states=None):

        N, T, *_ = inputs[0].shape
        if self.n_steps is None:
            n_steps = T
        else:
            n_steps = self.n_steps
        if self.batch_size is None:
            batch_size = N
        else:
            batch_size = self.batch_size
        if extra_inputs is None:
            extra_inputs = list()

        logger.log("Start training...")

        learning_rate = self.learning_rate

        best_loss = None
        best_params = None
        n_no_improvements = 0

        for epoch_id in range(self.n_epochs):
            logger.log("Epoch %d" % epoch_id)

            progbar = pyprind.ProgBar(N * T)
            losses = []
            diags = OrderedDict([(k, []) for k in self.diagnostic_vars.keys()])
            for batch_idx in range(0, N, batch_size):
                batch_sliced_inputs = [x[batch_idx:batch_idx + self.batch_size] for x in inputs]
                if prev_states is None:
                    states = np.tile(
                        self.rnn_init_state.eval().reshape((1, -1)),
                        (batch_sliced_inputs[0].shape[0], 1)
                    )
                else:
                    states = prev_states[batch_idx:batch_idx + self.batch_size]
                for t in range(0, T, n_steps):
                    time_sliced_inputs = [x[:, t:t + n_steps] for x in batch_sliced_inputs]
                    # The last input is the valid mask. Only bother computing if at least one entry is valid
                    if np.any(np.nonzero(time_sliced_inputs[-1])):
                        _, loss, states, *diagnostics = self.f_train(
                            *(time_sliced_inputs + extra_inputs + [states, learning_rate])
                        )
                        losses.append(loss)
                        for k, diag_val in zip(self.diagnostic_vars.keys(), diagnostics):
                            diags[k].append(diag_val)
                        progbar.update(len(states) * (min(t + n_steps, T) - t), force_flush=True)
                    else:
                        progbar.update(len(states) * (T - t), force_flush=True)
                        break
            if progbar.active:
                progbar.stop()

            diags = OrderedDict([(k, np.mean(vals)) for k, vals in diags.items()])
            loss = np.mean(losses)

            if self.verbose:
                log_message = "Loss: %f" % loss
                for k, v in diags.items():
                    log_message += "; %s: %f" % (k, v)
                logger.log(log_message)

            if best_loss is None or loss <= best_loss:
                best_loss = loss
                best_params = self.target.get_param_values()
                n_no_improvements = 0
            else:
                n_no_improvements += 1

            if self.callback or callback:
                callback_args = dict(
                    loss=loss,
                    diagnostics=diags,
                    itr=epoch_id,
                    best_loss=best_loss,
                    n_no_improvements=n_no_improvements,
                    learning_rate=learning_rate,
                )
                if self.callback is not None:
                    if not self.callback(callback_args):
                        return
                if callback is not None:
                    if not callback(**callback_args):
                        return

            if self.anneal_learning_rate:
                if n_no_improvements >= self.no_improvement_tolerance:
                    learning_rate *= self.anneal_ratio
                    n_no_improvements = 0
                    self.target.set_param_values(best_params)
