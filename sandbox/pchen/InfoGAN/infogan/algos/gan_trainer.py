from regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from distributions import Bernoulli, Gaussian
import sys

TINY = 1e-8


class GANTrainer(object):
    def __init__(self,
                 model,
                 dataset,
                 batch_size,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=10000,
                 discriminator_learning_rate=2e-4,
                 generator_learning_rate=2e-4,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []

    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, 28 * 28])

        with pt.defaults_scope(phase=pt.Phase.train):
            z_var = self.model.latent_dist.sample_prior(self.batch_size)
            fake_x, _ = self.model.generate(z_var)
            all_d = self.model.discriminate(tf.concat(0, [input_tensor, fake_x]))
            real_d = all_d[:self.batch_size, :]
            fake_d = all_d[self.batch_size:, :]

            discriminator_loss = - tf.reduce_mean(tf.log(real_d + TINY) + tf.log(1. - fake_d + TINY))
            generator_loss = - tf.reduce_mean(tf.log(fake_d + TINY))

            all_vars = tf.trainable_variables()
            d_vars = [var for var in all_vars if var.name.startswith('d_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]

            tf.scalar_summary("discriminator_loss", discriminator_loss)
            tf.scalar_summary("generator_loss", generator_loss)

            self.log_vars.append(("discriminator_loss", discriminator_loss))
            self.log_vars.append(("generator_loss", generator_loss))
            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))
            # self.log_vars.append(("min_z_var", tf.reduce_min(z_var)))
            # self.log_vars.append(("max_z_var", tf.reduce_max(z_var)))

            discriminator_optimizer = tf.train.AdamOptimizer(self.discriminator_learning_rate, beta1=0.5)
            self.discriminator_trainer = pt.apply_optimizer(discriminator_optimizer, losses=[discriminator_loss],
                                                            var_list=d_vars)

            generator_optimizer = tf.train.AdamOptimizer(self.generator_learning_rate, beta1=0.5)
            self.generator_trainer = pt.apply_optimizer(generator_optimizer, losses=[generator_loss], var_list=g_vars)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                # img_var = fake_x
                z_var = self.model.latent_dist.sample_prior(self.batch_size)
                img_var, _ = self.model.generate(z_var)
                # _, x_dist_info = self.model.generate(z_var)
                #
                # if isinstance(self.model.output_dist, Bernoulli):
                #    img_var = x_dist_info["p"]
                # elif isinstance(self.model.output_dist, Gaussian):
                #    img_var = x_dist_info["mean"]

                rows = 10
                img_var = tf.reshape(img_var, [self.batch_size, 28, 28, 1])
                img_var = img_var[:rows * rows, :, :, :]
                imgs = tf.reshape(img_var, [rows, rows, 28, 28, 1])
                stacked_img = []
                for row in xrange(rows):
                    row_img = []
                    for col in xrange(rows):
                        row_img.append(imgs[row, col, :, :, :])
                    stacked_img.append(tf.concat(1, row_img))
                imgs = tf.concat(0, stacked_img)
                imgs = tf.expand_dims(imgs, 0)
                tf.image_summary("image", imgs, max_images=3)

    def train(self):

        self.init_opt()

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            counter = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.batch_size)
                    # x = np.reshape(x, (-1, 28, 28, 1))
                    log_vals = sess.run([self.discriminator_trainer] + log_vars, {self.input_tensor: x})[1:]
                    sess.run(self.generator_trainer, {self.input_tensor: x})
                    all_log_vals.append(log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                x, _ = self.dataset.train.next_batch(self.batch_size)
                # x = np.reshape(x, (-1, 28, 28, 1))
                summary_str = sess.run(summary_op, {self.input_tensor: x})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))

                print(log_line)
                sys.stdout.flush()
