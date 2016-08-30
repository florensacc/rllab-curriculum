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


class CVVAE(VAE):

    def __init__(self,
                 model,
                 dataset,
                 batch_size,
                 alpha_update_interval=10,
                 **kwargs
    ):
        super(CVVAE, self).__init__(
            model,
            dataset,
            batch_size,
            **kwargs
        )
        self.alpha_update_interval = alpha_update_interval
        self.alpha_var = tf.Variable(
            initial_value=0.,
            trainable=True,
            name="cv_coeff",
        )
        self.alpha_trainer = None

    def init_hook(self, vars):
        from rllab.misc.ext import extract
        eval, final_losses, log_vars, init,\
            log_p_x_given_z, log_p_z, z_dist_info,\
            z_var, ndim = extract(
            vars,
            "eval", "final_losses", "log_vars", "init",
            "log_p_x_given_z", "log_p_z", "z_dist_info",
            "z_var", "ndim"
        )
        # if eval:
        #     return

        # redoing the objective for training phase
        final_losses[:] = []
        assert self.monte_carlo_kl
        assert self.min_kl == 0.
        q_z_ent = self.model.inference_dist.entropy(z_dist_info)
        ent_vlbs = log_p_x_given_z + log_p_z + q_z_ent
        cvs = q_z_ent + self.model.inference_dist.logli(
            z_var,
            dict([
                (k, tf.stop_gradient(v))
                for k, v in z_dist_info.items()
            ])
        )
        ent_vlb = tf.reduce_mean(ent_vlbs) / ndim
        cv = tf.reduce_mean(cvs) / ndim
        cv_ent_vlb = (ent_vlb - tf.stop_gradient(self.alpha_var) * cv) / ndim
        log_vars.append((
            "ent_vlb",
            ent_vlb
        ))
        log_vars.append((
            "cv",
            cv
        ))
        log_vars.append((
            "cv_ent_vlb_alpha1",
            ent_vlb - cv
        ))
        log_vars.append((
            "cv_ent_vlb",
            cv_ent_vlb
        ))
        log_vars.append((
            "alpha",
            self.alpha_var
        ))

        final_losses[:] = [-cv_ent_vlb, ]
        if init or eval:
            # during init, skip anything that has to do with grad
            return
        params = [
            p for p in tf.trainable_variables() if "cv_coeff" not in p.name
        ]
        total_params = 0
        for variable in params:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            # print(shape) print(len(shape))
            variable_parametes = 1
            for dim in shape:
                # print(dim)
                variable_parametes *= dim.value
            # print(variable_parametes)
            total_params += variable_parametes
        ent_grads = tf.gradients(ent_vlb, params)
        cv_grads = tf.gradients(cv, params)
        # import ipdb; ipdb.set_trace()
        grad_norm = tf.reduce_sum(
            [
                tf.nn.l2_loss(tf.stop_gradient(eg) - self.alpha_var*tf.stop_gradient(cg))
                for eg, cg in zip(ent_grads, cv_grads) if eg is not None and cg is not None
            ]
        ) / total_params

        with tf.variable_scope("optim_alpha"):
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = self.optimizer_cls(**self.optimizer_args)
            self.alpha_trainer = optimizer.minimize(grad_norm, var_list=[self.alpha_var])

    def iter_hook(self, sess, counter, feed, **kw):
        if (counter+1) % self.alpha_update_interval == 0:
            sess.run(
                [self.alpha_trainer],
                feed
            )
