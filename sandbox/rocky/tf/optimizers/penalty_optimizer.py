from rllab.misc.ext import lazydict
from rllab.misc import logger
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
import numpy as np
import scipy.optimize


class PenaltyOptimizer(Serializable):
    def __init__(
            self,
            optimizer,
            max_opt_itr=20,
            initial_penalty=1.0,
            min_penalty=1e-2,
            max_penalty=1e6,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            adapt_penalty=True,
            adapt_itr=32,
            data_split=None,
            max_penalty_itr=10,
            barrier_coeff=0.,
    ):
        Serializable.quick_init(self, locals())
        self._optimizer = optimizer
        self._max_opt_itr = max_opt_itr
        self._penalty = initial_penalty
        self._initial_penalty = initial_penalty
        self._min_penalty = min_penalty
        self._max_penalty = max_penalty
        self._increase_penalty_factor = increase_penalty_factor
        self._decrease_penalty_factor = decrease_penalty_factor
        self._adapt_penalty = adapt_penalty

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._min_constraint_val = None
        self._constraint_name = None
        self._adapt_itr = adapt_itr
        self._data_split = data_split
        self._max_penalty_itr = max_penalty_itr
        self._barrier_coeff = barrier_coeff

    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name="constraint", *args, **kwargs):
        """
        :param loss: Symbolic expression for the loss function.
        :param target: A parameterized object to optimize over. It should implement methods of the
        :class:`rllab.core.paramerized.Parameterized` class.
        :param leq_constraint: A constraint provided as a tuple (f, epsilon), of the form f(*inputs) <= epsilon.
        :param inputs: A list of symbolic variables as inputs
        :return: No return value.
        """
        constraint_term, constraint_value = leq_constraint
        penalty_var = tf.Variable(initial_value=self._initial_penalty, trainable=False, name='penalty_coeff')
        penalized_loss = loss + penalty_var * constraint_term + \
                         self._barrier_coeff * tf.cast(constraint_term > constraint_value, tf.float32) * \
                         (constraint_value - constraint_term) ** 2
        # -TT.log(constraint_value*1.3 - constraint_term)

        self._target = target
        self._max_constraint_val = constraint_value
        self._min_constraint_val = 0.8 * constraint_value
        self._constraint_name = constraint_name
        self._penalty_var = penalty_var

        def get_opt_output():
            flat_grad = tensor_utils.flatten_tensor_variables(tf.gradients(
                penalized_loss, target.get_params(trainable=True),
            ))
            return [tf.cast(penalized_loss, tf.float64), tf.cast(flat_grad, tf.float64)]

        self._opt_fun = lazydict(
            f_loss=lambda: tensor_utils.compile_function(inputs, loss, log_name="f_loss"),
            f_constraint=lambda: tensor_utils.compile_function(inputs, constraint_term, log_name="f_constraint"),
            f_penalized_loss=lambda: tensor_utils.compile_function(
                inputs=inputs,
                outputs=[penalized_loss, loss, constraint_term],
                log_name="f_penalized_loss",
            ),
            f_opt=lambda: tensor_utils.compile_function(
                inputs=inputs,
                outputs=get_opt_output(),
                log_name="f_opt"
            )
        )

        self._optimizer.update_opt(penalized_loss, target, inputs, *args, **kwargs)

    def loss(self, inputs):
        return self._opt_fun["f_loss"](*inputs)

    def constraint_val(self, inputs):
        return self._opt_fun["f_constraint"](*inputs)

    def optimize(self, inputs):

        ob_no = inputs[0]
        action_na = inputs[1]
        advantage_n = inputs[2]

        inputs = tuple(inputs)
        if self._data_split is not None:
            maxlen = len(inputs[0])
            cutoff = int(maxlen * self._data_split)
            val_inputs = tuple(
                inp[cutoff:] for inp in inputs
            )
            inputs = tuple(
                inp[:cutoff] for inp in inputs
            )
        f_penalized_loss = self._opt_fun["f_penalized_loss"]

        try_penalty = np.clip(
            tf.get_default_session().run(self._penalty_var), self._min_penalty, self._max_penalty)

        cur_params = self._target.get_param_values(trainable=True).astype('float64')

        train = []
        val = []
        _, init_try_loss, init_try_constraint_val = f_penalized_loss(*inputs)
        _, init_val_loss, init_val_constraint_val = f_penalized_loss(*val_inputs)
        train.append((init_try_loss, init_try_constraint_val))
        val.append((init_val_loss, init_val_constraint_val))
        logger.log('before optim penalty %f => loss %f (%f), %s %f (%f)' %
                   (try_penalty, init_try_loss, init_val_loss,
                    self._constraint_name, init_try_constraint_val, init_val_constraint_val))
        for penalty_itr in range(self._max_penalty_itr):
            tried_kl = try_penalty

            def dbp(itr, **kwargs):
                _, try_loss, try_constraint_val = f_penalized_loss(*inputs)
                _, val_loss, val_constraint_val = f_penalized_loss(*val_inputs)
                logger.log('[epoch %i] penalty %f => loss %f (%f), %s %f (%f)' %
                           (itr, try_penalty, try_loss, val_loss,
                            self._constraint_name, try_constraint_val, val_constraint_val))

            self._target.set_param_values(cur_params, trainable=True)
            self._optimizer.optimize(inputs, callback=dbp)

            # for _ in self._optimizer.optimize_gen(inputs, yield_itr=self._adapt_itr):
            # logger.log('trying penalty=%.3f...' % try_penalty)

            # _, try_loss, try_constraint_val = f_penalized_loss(*inputs)

            # logger.log('penalty %f => loss %f, %s %f' %
            #            (try_penalty, try_loss, self._constraint_name, try_constraint_val))
            if self._data_split is not None:
                _, try_loss, try_constraint_val = f_penalized_loss(*inputs)
                _, val_loss, val_constraint_val = f_penalized_loss(*val_inputs)
                train.append((try_loss, try_constraint_val))
                val.append((val_loss, val_constraint_val))
                logger.log('penalty %f => loss %f (%f), %s %f (%f)' %
                           (try_penalty, try_loss, val_loss,
                            self._constraint_name, try_constraint_val, val_constraint_val))

            if not self._adapt_penalty:
                break

            # Increase penalty if constraint violated, or if constraint term is NAN
            if try_constraint_val > self._max_constraint_val or np.isnan(try_constraint_val):
                penalty_scale_factor = self._increase_penalty_factor
            elif try_constraint_val <= self._min_constraint_val:
                # if constraint is lower than threshold, shrink penalty
                penalty_scale_factor = self._decrease_penalty_factor
            else:
                # if things are good, keep current penalty
                break
            try_penalty *= penalty_scale_factor
            try_penalty = np.clip(try_penalty, self._min_penalty, self._max_penalty)
            tf.get_default_session().run(
                tf.assign(self._penalty_var, try_penalty)
            )

        logger.record_tabular('KLCoeff', tried_kl)
        logger.record_tabular('TrainSrrReduction', init_try_loss - try_loss)
        logger.record_tabular('ValiSrrReduction', init_val_loss - val_loss)
