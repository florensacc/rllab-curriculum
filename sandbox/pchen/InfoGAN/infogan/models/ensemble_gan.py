from sandbox.pchen.InfoGAN.infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
import prettytensor as pt
import tensorflow as tf
import sandbox.pchen.InfoGAN.infogan.misc.custom_ops
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import leaky_rectify


class EnsembleGAN(object):
    def __init__(
            self,
            output_dist,
            latent_spec,
            batch_size,
            image_shape,
            network_type,
            nr_models=2,
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
        self.batch_size = batch_size
        self.network_type = network_type
        self.image_shape = image_shape

        image_size = image_shape[0]

        self.discriminator_templates = []

        if network_type == "face_conv":
            raise NotImplementedError
        elif network_type == "mnist":
            with tf.variable_scope("d_net"):
                for i in range(nr_models):
                    with tf.variable_scope("model%s"%i):
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
                        self.discriminator_templates.append(
                            shared_template.custom_fully_connected(1)
                        )

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
        elif network_type == "mnist_nobn":
            with tf.variable_scope("d_net"):
                for i in range(nr_models):
                    with tf.variable_scope("model%s"%i):
                        shared_template = \
                            (pt.template("input").
                             reshape([-1] + list(image_shape)).
                             custom_conv2d(64, k_h=4, k_w=4).
                             apply(leaky_rectify).
                             custom_conv2d(128, k_h=4, k_w=4).
                             apply(leaky_rectify).
                             custom_fully_connected(1024).
                             apply(leaky_rectify))
                        self.discriminator_templates.append(
                            shared_template.custom_fully_connected(1)
                        )

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
        elif network_type == "cifar1":
            with tf.variable_scope("d_net"):
                for i in range(nr_models):
                    with tf.variable_scope("model%s" % i):
                        base_filter = [4, 6, 12, 16][i % 4]
                        shared_template = \
                            (pt.template("input").
                             reshape([-1] + list(image_shape)).
                             custom_conv2d(base_filter, k_h=4, k_w=4).
                             apply(leaky_rectify).
                             custom_conv2d(base_filter * 2, k_h=4, k_w=4).
                             apply(leaky_rectify).
                             custom_conv2d(base_filter * 4, k_h=4, k_w=4).
                             # conv_batch_norm().
                             apply(leaky_rectify).
                             custom_fully_connected(base_filter * 16).
                             # fc_batch_norm().
                             apply(leaky_rectify)
                             )
                        self.discriminator_templates.append(
                            shared_template.custom_fully_connected(1)
                        )

            from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import resconv_v1, resdeconv_v1
            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(image_size // 8 * image_size // 8 * 512).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 8, image_size // 8, 512]).
                     custom_deconv2d([0, image_size // 4, image_size // 4, 256], k_h=5, k_w=5).
                     conv_batch_norm().
                     apply(tf.nn.relu))
                self.generator_template = (self.generator_template.
                                           custom_deconv2d([0, image_size // 2, image_size // 2, 128], k_h=5, k_w=5).
                                           conv_batch_norm().
                                           apply(tf.nn.relu))
                self.generator_template = (self.generator_template.
                                           custom_deconv2d([0] + list(image_shape), k_h=5, k_w=5).
                                           flatten())
        elif network_type == "cifar2":
            with tf.variable_scope("d_net"):
                for i in range(nr_models):
                    with tf.variable_scope("model%s"%i):
                        base_filter = [4, 6, 12, 16][i%4]
                        seq = [3,2,1,0][i%4]
                        tmp = pt.template("input").\
                            reshape([-1] + list(image_shape)).\
                            custom_conv2d(base_filter, k_h=4, k_w=4).\
                            conv_batch_norm().\
                            apply(leaky_rectify) # 16
                        for _ in range(seq):
                            tmp = tmp.custom_conv2d(
                                base_filter,
                                k_h=3, k_w=3,
                                d_h=1, d_w=1,
                            ).apply(
                                leaky_rectify
                            )
                        tmp = tmp.custom_conv2d(
                            base_filter*2,
                            k_h=3, k_w=3,
                        ).conv_batch_norm().apply(
                            leaky_rectify
                        ) # 8
                        for _ in range(seq):
                            tmp = tmp.custom_conv2d(
                                base_filter,
                                k_h=3, k_w=3,
                                d_h=1, d_w=1,
                            ).apply(
                                leaky_rectify
                            )
                        tmp = tmp.custom_conv2d(
                            base_filter*3,
                            k_h=3, k_w=3,
                        ).conv_batch_norm().apply(
                            leaky_rectify
                        ) # 4
                        for _ in range(seq):
                            tmp = tmp.custom_conv2d(
                                base_filter,
                                k_h=3, k_w=3,
                                d_h=1, d_w=1,
                            ).apply(
                                leaky_rectify
                            )
                        shared_template = \
                            (pt.template("input").
                             reshape([-1] + list(image_shape)).
                             custom_conv2d(base_filter, k_h=4, k_w=4).
                             apply(leaky_rectify).
                             custom_conv2d(base_filter*2, k_h=4, k_w=4).
                             apply(leaky_rectify).
                             custom_conv2d(base_filter*4, k_h=4, k_w=4).
                             # conv_batch_norm().
                             apply(leaky_rectify).
                             custom_fully_connected(base_filter*16).
                             # fc_batch_norm().
                             apply(leaky_rectify)
                        )
                        self.discriminator_templates.append(
                            shared_template.
                                custom_fully_connected(base_filter*16).
                                apply(leaky_rectify).
                                custom_fully_connected(1)
                        )

            from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import resconv_v1, resdeconv_v1
            with tf.variable_scope("g_net"):
                self.generator_template = \
                    (pt.template("input").
                     custom_fully_connected(image_size // 8 * image_size // 8 * 512).
                     fc_batch_norm().
                     apply(tf.nn.relu).
                     reshape([-1, image_size // 8, image_size // 8, 512]).
                     custom_deconv2d([0, image_size // 4, image_size // 4, 256], k_h=5, k_w=5).
                     conv_batch_norm().
                     apply(tf.nn.relu))
                # self.generator_template = resconv_v1(
                #     self.generator_template,
                #     4,
                #     64,
                #     stride=1,
                #     add_coeff=0.3
                # )
                self.generator_template = (self.generator_template.
                     custom_deconv2d([0, image_size // 2, image_size // 2, 128], k_h=5, k_w=5).
                     conv_batch_norm().
                     apply(tf.nn.relu))
                # self.generator_template = resconv_v1(
                #     self.generator_template,
                #     4,
                #     32,
                #     stride=1,
                #     add_coeff=0.3
                # )
                self.generator_template = (self.generator_template.
                     custom_deconv2d([0] + list(image_shape), k_h=5, k_w=5).
                     flatten())
        else:
            raise NotImplementedError

    def discriminate(self, x_var, logits=False):
        ds = [
            discriminator_template.construct(input=x_var).tensor[:, 0]
            for discriminator_template in self.discriminator_templates
        ] # nr_models x bs
        ds = tf.transpose(tf.convert_to_tensor(ds)) # bs x nr_models
        assert logits
        # d = tf.reduce_max(
        #     ds,
        #     reduction_indices=[0]
        # )
        mean_d, var_d = tf.nn.moments(ds, [1])
        tf.scalar_summary("mean_var_d", tf.reduce_mean(var_d))
        # if not logits:
        #     d = tf.nn.sigmoid(d)
        return ds

    def generate(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info

