from sandbox.pchen.InfoGAN.infogan.misc.distributions import Product, Distribution, Gaussian
import prettytensor as pt
import tensorflow as tf
# from deconv import deconv2d
import sandbox.pchen.InfoGAN.infogan.misc.custom_ops
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import leaky_rectify


class BayesRegularizedHelmholtzMachine(object):
    def __init__(
            self,
            output_dist,
            latent_spec,
            batch_size,
            image_shape,
            network_type,
            inference_dist=None,
            local_reparam=False,
            prior_std=0.05,
    ):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        if len(latent_spec) == 1:
            self.latent_dist = latent_spec[0][0]
        else:
            self.latent_dist = Product([x for x, _ in latent_spec])
        if inference_dist is None:
            inference_dist = self.latent_dist
        self.inference_dist = inference_dist
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape

        image_size = image_shape[0]

        with pt.defaults_scope(
                activation_fn=tf.nn.relu,
                local_reparam=local_reparam,
                prior_std=prior_std,
        ):
            if self.network_type == "mlp":
                self.encoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(self.inference_dist.dist_flat_dim, activation_fn=None))
                self.reg_encoder_template = \
                    (pt.template('input').
                     bayes_fully_connected(500).
                     bayes_fully_connected(self.reg_latent_dist.dist_flat_dim, activation_fn=None))

                self.decoder_template = \
                    (pt.template('input').
                     bayes_fully_connected(500).
                     bayes_fully_connected(self.output_dist.dist_flat_dim, activation_fn=None))
            elif self.network_type == "deep_mlp":
                self.encoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(self.inference_dist.dist_flat_dim, activation_fn=None))
                self.reg_encoder_template = \
                    (pt.template('input').
                     bayes_fully_connected(500).
                     bayes_fully_connected(500).
                     bayes_fully_connected(500).
                     bayes_fully_connected(self.reg_latent_dist.dist_flat_dim, activation_fn=None))
                self.decoder_template = \
                    (pt.template('input').
                     bayes_fully_connected(500).
                     bayes_fully_connected(500).
                     bayes_fully_connected(500).
                     bayes_fully_connected(self.output_dist.dist_flat_dim, activation_fn=None))
            else:
                raise NotImplementedError

    def encode(self, x_var):
        z_dist_flat = self.encoder_template.construct(input=x_var).tensor
        z_dist_info = self.inference_dist.activate_dist(z_dist_flat)
        return self.inference_dist.sample(z_dist_info), z_dist_info

    def reg_encode(self, x_var):
        reg_z_dist_flat = self.reg_encoder_template.construct(input=x_var).tensor
        reg_z_dist_info = self.reg_latent_dist.activate_dist(reg_z_dist_flat)
        return self.reg_latent_dist.sample(reg_z_dist_info), reg_z_dist_info

    def decode(self, z_var):
        x_dist_flat = self.decoder_template.construct(input=z_var).tensor
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)

        return self.output_dist.sample(x_dist_info), x_dist_info

    def reg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)
