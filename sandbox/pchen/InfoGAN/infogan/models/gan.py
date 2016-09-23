from sandbox.pchen.InfoGAN.infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import sandbox.pchen.InfoGAN.infogan.misc.custom_ops
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import leaky_rectify


class GAN(object):
    def __init__(
            self,
            output_dist,
            latent_spec,
            batch_size,
            image_shape,
            network_type,
            # use_separate_recog=False,
    ):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        # self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        # self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        # self.use_separate_recog = use_separate_recog
        # assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)
        #
        # self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        # self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        image_size = image_shape[0]
        if network_type == "face_conv":
            raise NotImplementedError
        elif network_type == "mnist":
            with tf.variable_scope("d_net"):
                shared_template = \
                    (pt.template("input").
                     reshape([-1] + list(image_shape)).
                     custom_conv2d(64, k_h=4, k_w=4).
                     apply(leaky_rectify).
                     custom_conv2d(128, k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(leaky_rectify))
                self.discriminator_template = shared_template.custom_fully_connected(1)
                # self.encoder_template = \
                #     (shared_template.
                #      custom_fully_connected(128).
                #      fc_batch_norm().
                #      apply(leaky_rectify).
                #      custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(image_size // 4 * image_size // 4 * 128).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 4, image_size // 4, 128]).
                     custom_deconv2d([0, image_size // 2, image_size // 2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())
        else:
            raise NotImplementedError

    def discriminate(self, x_var, logits=False):
        d = self.discriminator_template.construct(input=x_var)
        if not logits:
            d = tf.nn.sigmoid(d[:, 0])
        return d

    def generate(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info

