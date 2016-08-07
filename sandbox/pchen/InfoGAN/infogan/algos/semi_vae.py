from sandbox.pchen.InfoGAN.infogan.algos.vae import VAE
from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Bernoulli, Gaussian, Mixture
import rllab.misc.logger as logger
import sys
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, logsumexp


class SemiVAE(VAE):

    def __init__(self,
                 model,
                 dataset,
                 batch_size,
                 sup_batch_size,
                 sup_coeff,
                 **kwargs
    ):
        super(SemiVAE, self).__init__(
            model,
            dataset,
            batch_size,
            **kwargs
        )
        self.sup_coeff = sup_coeff
        self.sup_batch_size = sup_batch_size
        self.label_dim = 10
        with pt.defaults_scope(
                activation_fn=tf.nn.elu,
        ):
            self.classfication_template = (
                pt.template('input').
                fully_connected(30).
                fully_connected(self.label_dim, activation_fn=None)
            )

    def init_hook(self, vars):
        from rllab.misc.ext import extract
        eval, final_losses, log_vars = extract(
            vars,
            "eval", "final_losses", "log_vars"
        )
        if eval:
            self.eval_label_tensor = \
                sup_label_tensor = \
                tf.placeholder(
                    tf.float32,
                    [self.eval_batch_size, self.label_dim],
                    "eval_label"
                )
            sup_z = vars["z_var"]
        else:
            self.sup_input_tensor = \
                sup_input_tensor = \
                tf.placeholder(
                    tf.float32,
                    [self.sup_batch_size, self.dataset.image_dim],
                    "sup_input",
                )
            self.sup_label_tensor = \
                sup_label_tensor = \
                tf.placeholder(
                    tf.float32,
                    [self.sup_batch_size, self.label_dim],
                    "sup_label"
                )
            sup_z, _, _ = self.model.encode(sup_input_tensor, k=1)

        sup_logits = self.classfication_template.construct(input=sup_z).tensor
        sup_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(sup_logits, sup_label_tensor)
        )
        log_vars += [
            ("sup_cross_ent", sup_loss),
            ("accuracy", tf.reduce_mean(
                tf.cast(
                    tf.equal(tf.argmax(sup_logits, 1), tf.argmax(sup_label_tensor, 1)),
                    tf.float32,
                )
            ))
        ]
        final_losses.append(sup_loss * self.sup_coeff)

    def prepare_feed(self, data, bs):
        x, _ = data.next_batch(bs)
        assert self.weight_redundancy == 1
        sx, sy = self.dataset.supervised_train.next_batch(self.sup_batch_size)
        return {
            self.input_tensor: x,
            self.sup_input_tensor: sx,
            self.sup_label_tensor: sy,
        }

    def prepare_eval_feed(self, data, bs):
        x, y = data.next_batch(bs)
        x = np.tile(x, [self.weight_redundancy, 1])
        y = np.tile(y, [self.weight_redundancy, 1])
        return {
            self.eval_input_tensor: x,
            self.eval_label_tensor: y,
        }
