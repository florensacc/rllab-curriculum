from sandbox.pchen.InfoGAN.infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import sandbox.pchen.InfoGAN.infogan.misc.custom_ops
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import leaky_rectify


class RegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, batch_size, image_shape, network_type, use_separate_recog=False):
        """
        :type output_dist: Distribution
        :type latent_spec: list[(Distribution, bool)]
        :type batch_size: int
        :type network_type: string
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape
        self.use_separate_recog = use_separate_recog
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import resize_nearest_neighbor

        image_size = image_shape[0]
        if network_type == "face_conv":
            raise NotImplementedError
        elif network_type == "tmp1_subg_convd":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (
                    pt.template("input").
                        reshape([-1] + list(image_shape)).
                        custom_conv2d(64, k_h=4, k_w=4, d_h=2, d_w=2).
                        apply(leaky_rectify).
                        custom_conv2d(128, k_h=4, k_w=4, d_h=2, d_w=2).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=2, d_w=2).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_fully_connected(1024).
                        apply(leaky_rectify).
                        custom_fully_connected(dim))
                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)

                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(256 * image_size // 8 * image_size // 8).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 8, image_size // 8, 256]).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256],
                        k_h=4, k_w=4, d_h=1, d_w=1
                    ).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256],
                        k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                     # apply(resize_nearest_neighbor, 2).
                     conv2d_mod(2, 256*4, activation_fn=None).
                     depool2d_split().
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_conv2d(64, k_h=4, k_w=4, d_h=1, d_w=1).
                     # apply(resize_nearest_neighbor, 2).
                     conv2d_mod(2, 64*4, activation_fn=None).
                     depool2d_split().
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     # custom_conv2d(image_shape[-1], k_h=4, k_w=4, d_h=1, d_w=1).
                     # apply(resize_nearest_neighbor, 2).
                     conv2d_mod(2, image_shape[-1]*4, activation_fn=None).
                     depool2d_split().
                     flatten())
        elif network_type == "tmp1_subg_resized":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (
                    pt.template("input").
                        reshape([-1] + list(image_shape)).
                        custom_conv2d(64, k_h=4, k_w=4, d_h=1, d_w=1).
                        apply(resize_nearest_neighbor, 0.5).
                        apply(leaky_rectify).
                        custom_conv2d(128, k_h=4, k_w=4, d_h=1, d_w=1).
                        apply(resize_nearest_neighbor, 0.5).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        apply(resize_nearest_neighbor, 0.5).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_fully_connected(1024).
                        apply(leaky_rectify).
                        custom_fully_connected(dim))
                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)

                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(256 * image_size // 8 * image_size // 8).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 8, image_size // 8, 256]).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256],
                        k_h=4, k_w=4, d_h=1, d_w=1
                    ).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256],
                        k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                     # apply(resize_nearest_neighbor, 2).
                     conv2d_mod(2, 256*4, activation_fn=None).
                     depool2d_split().
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_conv2d(64, k_h=4, k_w=4, d_h=1, d_w=1).
                     # apply(resize_nearest_neighbor, 2).
                     conv2d_mod(2, 64*4, activation_fn=None).
                     depool2d_split().
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     # custom_conv2d(image_shape[-1], k_h=4, k_w=4, d_h=1, d_w=1).
                     # apply(resize_nearest_neighbor, 2).
                     conv2d_mod(2, image_shape[-1]*4, activation_fn=None).
                     depool2d_split().
                     flatten())
        elif network_type == "tmp1_resize":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (
                    pt.template("input").
                        reshape([-1] + list(image_shape)).
                        custom_conv2d(64, k_h=4, k_w=4, d_h=1, d_w=1).
                        apply(resize_nearest_neighbor, 0.5).
                        apply(leaky_rectify).
                        custom_conv2d(128, k_h=4, k_w=4, d_h=1, d_w=1).
                        apply(resize_nearest_neighbor, 0.5).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        apply(resize_nearest_neighbor, 0.5).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                        conv_batch_norm().
                        apply(leaky_rectify).
                        custom_fully_connected(1024).
                        apply(leaky_rectify).
                        custom_fully_connected(dim))
                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)

                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(256 * image_size // 8 * image_size // 8).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 8, image_size // 8, 256]).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256],
                        k_h=4, k_w=4, d_h=1, d_w=1
                     ).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256],
                        k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                     apply(resize_nearest_neighbor, 2).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_conv2d(64, k_h=4, k_w=4, d_h=1, d_w=1).
                     apply(resize_nearest_neighbor, 2).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     # custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     custom_conv2d(image_shape[-1], k_h=4, k_w=4, d_h=1, d_w=1).
                     apply(resize_nearest_neighbor, 2).
                     flatten())
        elif network_type == "tmp1":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (pt.template("input").
                    reshape([-1] + list(image_shape)).
                    custom_conv2d(64, k_h=4, k_w=4).
                    apply(leaky_rectify).
                    custom_conv2d(128, k_h=4, k_w=4).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(256, k_h=4, k_w=4).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(256, k_h=4, k_w=4, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_fully_connected(1024).
                    apply(leaky_rectify).
                    custom_fully_connected(dim))
                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)

                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(256 * image_size // 8 * image_size // 8).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 8, image_size // 8, 256]).
                     custom_deconv2d(
                        [0, image_size//8, image_size//8, 256], k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//8, image_size//8, 256], k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//4, image_size//4, 128], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//2, image_size//2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())

        elif network_type == "mnist_semi":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (pt.template("input").
                    reshape([-1] + list(image_shape)).
                    custom_conv2d(64, k_h=4, k_w=4).
                    apply(leaky_rectify).
                    custom_conv2d(128, k_h=4, k_w=4).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(128, k_h=4, k_w=4, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(128, k_h=4, k_w=4, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_fully_connected(1024).
                    apply(leaky_rectify).
                    custom_fully_connected(dim))
                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)
                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)
            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(128 * image_size//4 * image_size//4).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size//4, image_size//4, 128]).
                     custom_deconv2d([0, image_size//4, image_size//4, 128], k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//4, image_size//4, 128], k_h=4, k_w=4, d_h=1, d_w=1).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//2, image_size//2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())
        elif network_type == "mnist_catgan":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (pt.template("input").
                    reshape([-1] + list(image_shape)).
                    custom_conv2d(32, k_h=5, k_w=5, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    max_pool(kernel=3, stride=2).
                    custom_conv2d(64, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(64, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    max_pool(kernel=3, stride=2).
                    custom_conv2d(128, k_h=3, k_w=3, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(10, k_h=1, k_w=1, d_h=1, d_w=1).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_fully_connected(128).
                    fc_batch_norm().
                    apply(leaky_rectify).
                    custom_fully_connected(dim))
                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)
                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)
            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     #custom_fully_connected(96 * 8 * 8).
                     #fc_batch_norm().
                     #apply(tf.nn.relu).
                     custom_fully_connected(96 * image_size//4 * image_size//4).
                     fc_batch_norm().
                     apply(leaky_rectify).
                     reshape([-1, image_size//4, image_size//4, 96]).
                     custom_deconv2d([0, image_size//2, image_size//2, 64], k_h=5, k_w=5).
                     conv_batch_norm().
                     apply(leaky_rectify).
                     custom_deconv2d([0, image_size, image_size, 64], k_h=5, k_w=5).
                     conv_batch_norm().
                     apply(leaky_rectify).
                     custom_deconv2d([0, image_size, image_size, 1], k_h=5, k_w=5, d_h=1, d_w=1).
                     flatten())

        elif network_type == "tmp":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (pt.template("input").
                    reshape([-1] + list(image_shape)).
                    custom_conv2d(64, k_h=4, k_w=4).
                    apply(leaky_rectify).
                    custom_conv2d(128, k_h=4, k_w=4).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_conv2d(256, k_h=4, k_w=4).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    # custom_conv2d(512, k_h=4, k_w=4).
                    # conv_batch_norm().
                    # apply(leaky_rectify).
                    custom_fully_connected(1024).
                    apply(leaky_rectify).
                    custom_fully_connected(dim))

                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)

                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)

                    # self.continuous_encoder_template = None
                    # self.discrete_encoder_template = None

            with tf.variable_scope("g_net"):
                # if use_separate_recog:
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(256 * image_size//8 * image_size//8).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size//8, image_size//8, 256]).
                     # custom_deconv2d([0, image_size//8, image_size//8, 256], k_h=4, k_w=4).
                     # conv_batch_norm().
                     # apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//4, image_size//4, 128], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0, image_size//2, image_size//2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())
        elif network_type == "large_conv":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: (pt.template("input").
                    reshape([-1] + list(image_shape)).
                    custom_conv2d(64, k_h=4, k_w=4).
                    apply(leaky_rectify).
                    custom_conv2d(128, k_h=4, k_w=4).
                    conv_batch_norm().
                    apply(leaky_rectify).
                    custom_fully_connected(1024).
                    apply(leaky_rectify).
                    custom_fully_connected(dim))

                if use_separate_recog:
                    # use separate networks for the discriminator + continuous encoder, and discrete encoder
                    self.discriminator_template = common_tmpl(1 + self.reg_cont_latent_dist.dist_flat_dim)
                    # self.continuous_encoder_template = common_tmpl()
                    self.discrete_encoder_template = common_tmpl(self.reg_disc_latent_dist.dist_flat_dim)

                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)

                    # self.continuous_encoder_template = None
                    # self.discrete_encoder_template = None

            with tf.variable_scope("g_net"):
                # if use_separate_recog:
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(1024).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(128 * image_size//4 * image_size//4).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size//4, image_size//4, 128]).
                     custom_deconv2d([0, image_size//2, image_size//2, 64], k_h=4, k_w=4).
                     conv_batch_norm().
                     apply(tf.nn.relu).
                     custom_deconv2d([0] + list(image_shape), k_h=4, k_w=4).
                     flatten())
        elif network_type == "mlp":
            with tf.variable_scope("d_net"):
                common_tmpl = lambda dim: \
                    (pt.template("input").
                     custom_fully_connected(500).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(dim))
                if use_separate_recog:
                    raise NotImplementedError # need to update the code
                    self.discriminator_template = common_tmpl(1)
                    self.recog_template = common_tmpl(self.reg_latent_dist.dist_flat_dim)
                else:
                    self.discriminator_template = common_tmpl(1 + self.reg_latent_dist.dist_flat_dim)
                    self.recog_template = None

            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(500).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     custom_fully_connected(self.output_dist.dist_flat_dim))
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
                self.encoder_template = \
                    (shared_template.
                     custom_fully_connected(128).
                     fc_batch_norm().
                     apply(leaky_rectify).
                     custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

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
        d_out = self.discriminator_template.construct(input=x_var)
        d = d_out[:, 0]
        if logits:
            d = tf.nn.sigmoid(d)

        if self.use_separate_recog:
            continuous_code = d_out[:, 1:self.reg_cont_latent_dist.dist_flat_dim+1]
            discrete_code = self.discrete_encoder_template.construct(input=x_var)
            # combine both to a single code
            cont_dist_flats = self.reg_cont_latent_dist.split_dist_flat(continuous_code)
            disc_dist_flats = self.reg_disc_latent_dist.split_dist_flat(discrete_code)
            cont_idx = 0
            disc_idx = 0
            dist_flats = []
            for dist in self.reg_latent_dist.dists:
                if isinstance(dist, Gaussian):
                    dist_flats.append(cont_dist_flats[cont_idx])
                    cont_idx += 1
                elif isinstance(dist, (Categorical, Bernoulli)):
                    dist_flats.append(disc_dist_flats[disc_idx])
                    disc_idx += 1
                else:
                    raise NotImplementedError
            reg_dist_flat = tf.concat(1, dist_flats)
        else:
            reg_dist_flat = d_out[:, 1:]
        reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
        return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info

    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

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
