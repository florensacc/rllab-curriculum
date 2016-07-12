from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf
from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab.misc import logger


class QuadraticPenaltyOptimizer(object):

    def __init__(self, batch_size_seq=None, epoch_seq=None, penalty_seq=None):
        if batch_size_seq is None:
            batch_size_seq = [128, None, None, None, None]
        if epoch_seq is None:
            epoch_seq = [50, 10, 10, 10, 10]
        if penalty_seq is None:
            penalty_seq = [10, 100, 1000, 10000, 100000]
        self.batch_size_seq = batch_size_seq
        self.epoch_seq = epoch_seq
        self.penalty_seq = penalty_seq

    def update_opt(self, loss, target, leq_constraint, inputs, constraint_name):
        constraint_term, max_constraint_value = leq_constraint
        f_loss = tensor_utils.compile_function(
            inputs=inputs,
            outputs=loss,
        )
        f_constraint = tensor_utils.compile_function(
            inputs=inputs,
            outputs=constraint_term,
        )
        self.max_constraint_value = max_constraint_value
        self.f_loss = f_loss
        self.f_constraint = f_constraint
        penalty_var = tf.placeholder(tf.float32, shape=tuple(), name="penalty")

        pen_loss = loss + penalty_var * tf.square(tf.maximum(0., constraint_term - max_constraint_value))

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(pen_loss, var_list=target.get_params(trainable=True))

        f_train = tensor_utils.compile_function(
            inputs=inputs + [penalty_var],
            outputs=[train_op, loss, constraint_term]
        )

        self.f_train = f_train
        self.target = target

    def loss(self, input_vals):
        return self.f_loss(*input_vals)

    def constraint_val(self, input_vals):
        return self.f_constraint(*input_vals)

    def optimize(self, input_vals):
        best_loss = None
        best_params = None
        for penalty, n_epoch, batch_size in zip(self.penalty_seq, self.epoch_seq, self.batch_size_seq):
            dataset = BatchDataset(inputs=input_vals, batch_size=batch_size)
            for epoch in xrange(n_epoch):
                for batch in dataset.iterate():
                    self.f_train(*(tuple(batch) + (penalty,)))
                cur_loss = self.loss(input_vals)
                cur_constraint = self.constraint_val(input_vals)
                if cur_constraint < self.max_constraint_value:
                    if best_loss is None or cur_loss < best_loss:
                        best_loss = cur_loss
                        best_params = self.target.get_param_values()
                logger.log("Loss: %f; KL: %f" % (self.loss(input_vals), self.constraint_val(input_vals)))
            # if constraint already satisfied, no need to continue
            if best_params is not None:
                break
        if best_params is not None:
            self.target.set_param_values(best_params)
