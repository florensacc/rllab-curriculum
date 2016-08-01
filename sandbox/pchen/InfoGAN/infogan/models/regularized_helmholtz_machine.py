from sandbox.pchen.InfoGAN.infogan.misc.distributions import Product, Distribution
import prettytensor as pt
import tensorflow as tf
# from deconv import deconv2d
import sandbox.pchen.InfoGAN.infogan.misc.custom_ops
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import leaky_rectify


class RegularizedHelmholtzMachine(object):
    def __init__(self, output_dist, latent_spec, batch_size,
            image_shape, network_type, inference_dist=None, wnorm=False):
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
        self.wnorm = wnorm
        self.book = pt.bookkeeper_for_default_graph()
        self.init_mode()

        image_size = image_shape[0]

        with pt.defaults_scope(activation_fn=tf.nn.relu):
            if self.network_type == "large_conv":
                self.encoder_template = \
                    (pt.template("input").
                     reshape([-1] + list(image_shape)).
                     custom_conv2d(64, k_h=4, k_w=4).
                     apply(leaky_rectify).
                     custom_conv2d(128, k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(self.latent_dist.dist_flat_dim))
                self.reg_encoder_template = \
                    (pt.template("input").
                     reshape([-1] + list(image_shape)).
                     custom_conv2d(64, k_h=4, k_w=4).
                     apply(leaky_rectify).
                     custom_conv2d(128, k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(self.reg_latent_dist.dist_flat_dim))
                self.decoder_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(128 * image_size / 4 * image_size / 4).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size / 4, image_size / 4, 128]).
                     custom_deconv2d([0, image_size / 2, image_size / 2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size, image_size, 1], k_h=4, k_w=4).
                     flatten())
            elif self.network_type == "mlp":
                self.encoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(self.inference_dist.dist_flat_dim, activation_fn=None))
                self.reg_encoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(self.reg_latent_dist.dist_flat_dim, activation_fn=None))
                self.decoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(self.output_dist.dist_flat_dim, activation_fn=None))
            elif self.network_type == "deep_mlp":
                self.encoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(self.inference_dist.dist_flat_dim, activation_fn=None))
                self.reg_encoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(self.reg_latent_dist.dist_flat_dim, activation_fn=None))
                self.decoder_template = \
                    (pt.template('input').
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(500).
                     fully_connected(self.output_dist.dist_flat_dim, activation_fn=None))
            elif self.network_type == "small_res":
                from prettytensor import UnboundVariable
                with pt.defaults_scope(activation_fn=tf.nn.elu, data_init=UnboundVariable('data_init'), wnorm=self.wnorm):
                    self.encoder_template = \
                        (pt.template('input', self.book).
                         reshape([self.batch_size] + list(image_shape)).
                         conv2d_mod(5, 16, ).
                         conv2d_mod(5, 16, residual=False).#True).
                         conv2d_mod(5, 16, residual=False).#True).
                         conv2d_mod(5, 32, stride=2). # out 32*14*14
                         conv2d_mod(5, 32, residual=False).#True).
                         conv2d_mod(5, 32, residual=False).#True).
                         conv2d_mod(5, 32, stride=2). # out 32*7*7
                         conv2d_mod(5, 32, residual=False).#True).
                         conv2d_mod(5, 32, residual=False).#True).
                         flatten().
                         wnorm_fc(450, ).
                         wnorm_fc(self.inference_dist.dist_flat_dim, activation_fn=None)
                         )
                    self.decoder_template = \
                        (pt.template('input', self.book).
                         wnorm_fc(450, ).
                         wnorm_fc(1568, ).
                         # fully_connected(450, ).
                         # fully_connected(1568, ).
                         reshape([self.batch_size, 7, 7, 32]).
                         # reshape([self.batch_size, 1, 1, 450]).
                         # custom_deconv2d([0] + [7,7,32], k_h=1, k_w=1).
                         conv2d_mod(5, 32, residual=False).#True).
                         conv2d_mod(5, 32, residual=False).#True).
                         custom_deconv2d([0] + [14,14,32], k_h=5, k_w=5).
                         conv2d_mod(5, 32, residual=False).#True).
                         conv2d_mod(5, 32, residual=False).#True).
                         custom_deconv2d([0] + [28,28,16], k_h=5, k_w=5).
                         conv2d_mod(5, 16, residual=False).#True).
                         conv2d_mod(5, 16, residual=False).#True).
                         conv2d_mod(5, 1, activation_fn=None).
                         flatten()
                         )
                    self.reg_encoder_template = \
                        (pt.template('input').
                         reshape([self.batch_size] + list(image_shape)).
                         custom_conv2d(5, 32, ).
                         custom_conv2d(5, 64, ).
                         custom_conv2d(5, 128, edges='VALID').
                         # dropout(0.9).
                         flatten().
                         fully_connected(self.reg_latent_dist.dist_flat_dim, activation_fn=None))
            elif self.network_type == "small_conv":
                self.encoder_template = \
                    (pt.template('input').
                     reshape([self.batch_size] + list(image_shape)).
                     custom_conv2d(5, 32, ).
                     custom_conv2d(5, 64, ).
                     custom_conv2d(5, 128, edges='VALID').
                     # dropout(0.9).
                     flatten().
                     fully_connected(self.latent_dist.dist_flat_dim, activation_fn=None))
                self.reg_encoder_template = \
                    (pt.template('input').
                     reshape([self.batch_size] + list(image_shape)).
                     custom_conv2d(5, 32, ).
                     custom_conv2d(5, 64, ).
                     custom_conv2d(5, 128, edges='VALID').
                     # dropout(0.9).
                     flatten().
                     fully_connected(self.reg_latent_dist.dist_flat_dim, activation_fn=None))
                self.decoder_template = \
                    (pt.template('input').
                     reshape([self.batch_size, 1, 1, self.latent_dist.dim]).
                     custom_deconv2d(3, 128, d_h=1, d_w=1, padding='VALID').
                     custom_deconv2d(5, 64, d_h=1, d_w=1, padding='VALID').
                     custom_deconv2d(5, 32, ).
                     custom_deconv2d(5, 1, activation_fn=None).
                     flatten())
            else:
                raise NotImplementedError

    def init_mode(self):
        self.data_init = True
        if self.book.summary_collections:
            self.book_summary_collections = self.book.summary_collections
            self.book.summary_collections = None

    def train_mode(self):
        self.data_init = False
        self.book.summary_collections = self.book_summary_collections

    def encode(self, x_var):
        z_dist_flat = self.encoder_template.construct(input=x_var, data_init=self.data_init).tensor
        z_dist_info = self.inference_dist.activate_dist(z_dist_flat)
        return self.inference_dist.sample(z_dist_info), z_dist_info

    def reg_encode(self, x_var):
        reg_z_dist_flat = self.reg_encoder_template.construct(input=x_var, ).tensor
        reg_z_dist_info = self.reg_latent_dist.activate_dist(reg_z_dist_flat)
        return self.reg_latent_dist.sample(reg_z_dist_info), reg_z_dist_info

    def decode(self, z_var):
        x_dist_flat = self.decoder_template.construct(input=z_var, data_init=self.data_init).tensor
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
