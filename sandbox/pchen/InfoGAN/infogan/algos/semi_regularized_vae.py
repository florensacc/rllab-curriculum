from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Bernoulli, Gaussian, Mixture
import sys


class RegularizedVAE(object):
    def __init__(self,
                 model,
                 dataset,
                 batch_size,
                 semi_batch_size,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=10000,
                 use_info_reg=True,
                 info_reg_coeff=1.0,
                 use_recog_reg=True,
                 recog_reg_coeff=1.0,
                 use_separate_recog=False,
                 learning_rate=1e-3,
                 summary_interval=100,
                 monte_carlo_kl=False,
                 min_kl=0.,
                 use_prior_reg=False,
                 weight_redundancy=1,
                 bnn_decoder=False,
                 bnn_kl_coeff=1.,
    ):
        """
        :type model: RegularizedHelmholtzMachine
        :type use_info_reg: bool
        :type info_reg_coeff: float
        :type use_recog_reg: bool
        :type recog_reg_coeff: float
        :type learning_rate: float
        """
        self.sample_zs = []
        self.sample_imgs = []
        self.model = model
        self.dataset = dataset
        self.true_batch_size = batch_size
        self.semi_batch_size = semi_batch_size
        self.batch_size = batch_size * weight_redundancy
        self.weight_redundancy = weight_redundancy
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.summary_interval = summary_interval
        self.use_info_reg = use_info_reg
        self.use_recog_reg = use_recog_reg
        self.info_reg_coeff = info_reg_coeff
        self.recog_reg_coeff = recog_reg_coeff
        self.use_separate_recog = use_separate_recog
        self.learning_rate = learning_rate
        self.trainer = None
        self.input_tensor = None
        self.semi_input_tensor = None
        self.semi_label_tensor = None
        self.log_vars = []
        self.monte_carlo_kl = monte_carlo_kl
        self.min_kl = min_kl
        self.use_prior_reg = use_prior_reg
        self.param_dist = None
        self.bnn_decoder = bnn_decoder
        self.bnn_kl_coeff = bnn_kl_coeff
        self.sess = None
        self.mean_var, self.std_var = None, None
        self.saved_prior_mean = None
        self.saved_prior_std = None

    def init_opt(self, optimizer=True):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])
        self.semi_input_tensor = semi_input_tensor = \
            tf.placeholder(tf.float32, [self.semi_batch_size, self.dataset.image_dim])
        self.semi_label_tensor = semi_label_tensor = \
            tf.placeholder(tf.float32, [self.semi_batch_size, 10])

        with pt.defaults_scope(phase=pt.Phase.train):
            z_var, z_dist_info = self.model.encode(input_tensor)
            x_var, x_dist_info = self.model.decode(z_var)

            log_p_x_given_z = self.model.output_dist.logli(input_tensor, x_dist_info)

            if self.monte_carlo_kl:
                # Construct the variational lower bound
                kl = tf.reduce_mean(
                    - self.model.latent_dist.logli_prior(z_var)  \
                    + self.model.inference_dist.logli(z_var, z_dist_info)
                )
            else:
                # Construct the variational lower bound
                kl = tf.reduce_mean(self.model.latent_dist.kl_prior(z_dist_info))

            ndim = self.model.output_dist.effective_dim

            true_vlb = tf.reduce_mean(log_p_x_given_z) - kl
            vlb = tf.reduce_mean(log_p_x_given_z) - tf.maximum(kl, self.min_kl * ndim)

            ld = self.model.latent_dist
            prior_z_var = ld.sample_prior(self.batch_size)
            prior_reg = tf.reduce_mean(
                ld.logli_prior(prior_z_var) - \
                    ld.logli_init_prior(prior_z_var)
                    # ld.logli(prior_z_var, ld.init_prior_dist_info(self.batch_size))
            )
            emp_prior_reg = tf.reduce_mean(
                ld.logli_prior(z_var) - \
                ld.logli_init_prior(z_var)
                # ld.logli(prior_z_var, ld.init_prior_dist_info(self.batch_size))
            )
            self.log_vars.append(("prior_reg", prior_reg))
            self.log_vars.append(("emp_prior_reg", emp_prior_reg))
            if self.use_prior_reg:
                vlb -= prior_reg

            sess = tf.Session()
            self.sess = sess
            with sess.as_default():
                init = tf.initialize_all_variables()
                sess.run(init)
                if self.bnn_decoder:
                    dist_vars = tf.get_collection("dist_vars")
                    mean_var, std_var = [
                        tf.concat(0, [
                            tf.reshape(var, [-1]) for var in vars
                            ]) for vars in zip(*dist_vars)
                        ]
                    pdim = int(mean_var.get_shape()[0])
                    mean_val, std_val = sess.run([mean_var, std_var])
                    self.saved_prior_mean = tf.Variable(
                        initial_value=mean_val,
                        trainable=False,
                    )
                    self.saved_prior_std = tf.Variable(
                        initial_value=std_val,
                        trainable=False,
                    )
                    self.param_dist = Gaussian(
                        pdim,
                        prior_mean=self.saved_prior_mean,
                        prior_stddev=self.saved_prior_std,
                    )
                    # self.dist_vars = dist_vars
                    self.mean_var, self.std_var = mean_var, std_var
                    # cur_dist_info = self.param_dist.activate_dist(
                    #     tf.reshape(tf.concat(0, [mean_var, tf.log(std_var)]), [1,-1]))
                    cur_dist_info = dict(
                        mean=tf.reshape(mean_var, [1,-1]),
                        stddev=tf.reshape(std_var, [1,-1]),
                    )
                    bnn_kl = tf.reduce_mean(self.param_dist.kl_prior(cur_dist_info))
                    vlb -= bnn_kl * self.bnn_kl_coeff
                    # tf.scalar_summary("bnn_kl", bnn_kl)
                    self.log_vars.append(("bnn_kl", bnn_kl))

            surr_vlb = vlb + tf.reduce_mean(
                    tf.stop_gradient(log_p_x_given_z) * self.model.latent_dist.nonreparam_logli(z_var, z_dist_info)
                )
            # Normalize by the dimensionality of the data distribution
            true_vlb /= ndim
            vlb /= ndim
            surr_vlb /= ndim
            kl /= ndim

            loss = - vlb
            surr_loss = - surr_vlb

            # tf.scalar_summary("vlb", vlb)
            # tf.scalar_summary("kl", kl)
            # tf.scalar_summary("true_vlb", true_vlb)
            tf.scalar_summary("surr_vlb", surr_vlb)

            self.log_vars.append(("vlb", vlb))
            self.log_vars.append(("kl", kl))
            self.log_vars.append(("true_vlb", true_vlb))
            for name, var in self.log_vars:
                tf.scalar_summary(name, var)

            if self.use_info_reg:
                # Compute the information regularization term, given by
                # I(c,G(z,c)) = H(c) - H(c|x)
                #             = E_{x,c} [log p(c|x) - log p(c)]
                sleep_z = self.model.latent_dist.sample_prior(self.batch_size)
                sleep_reg_z = self.model.reg_z(sleep_z)
                sleep_x, sleep_x_dist_info = self.model.decode(sleep_z)
                if self.use_separate_recog:
                    _, sleep_reg_z_dist_info = self.model.reg_encode(sleep_x)
                else:
                    _, sleep_z_dist_info = self.model.encode(sleep_x)
                    sleep_reg_z_dist_info = self.model.reg_dist_info(sleep_z_dist_info)
                log_q_c_given_x = self.model.reg_latent_dist.logli(sleep_reg_z, sleep_reg_z_dist_info)
                log_q_c = self.model.reg_latent_dist.logli_prior(sleep_reg_z)

                contrib = log_q_c_given_x - log_q_c

                info_reg_term = tf.reduce_mean(contrib)
                # The surrogate term needs to take into account the non-reparameterizable part and apply the score
                # function to it
                surr_info_reg_term = info_reg_term + tf.reduce_mean(
                    tf.stop_gradient(contrib) * self.model.output_dist.nonreparam_logli(sleep_x, sleep_x_dist_info)
                )

                info_reg_term /= self.model.reg_latent_dist.effective_dim
                surr_info_reg_term /= self.model.reg_latent_dist.effective_dim

                loss += - self.info_reg_coeff * info_reg_term
                surr_loss += - self.info_reg_coeff * surr_info_reg_term

                tf.scalar_summary("MI", info_reg_term)
                self.log_vars.append(("MI", info_reg_term))

            if self.use_recog_reg:
                if self.use_separate_recog:
                    wake_reg_z, wake_reg_z_dist_info = self.model.reg_encode(input_tensor)
                else:
                    wake_z, wake_z_dist_info = self.model.encode(input_tensor)
                    wake_reg_z = self.model.reg_z(wake_z)
                    wake_reg_z_dist_info = self.model.reg_dist_info(wake_z_dist_info)
                log_q_c_given_x = self.model.reg_latent_dist.logli(wake_reg_z, wake_reg_z_dist_info)
                log_q_c = self.model.reg_latent_dist.marginal_logli(wake_reg_z, wake_reg_z_dist_info)
                contrib = log_q_c_given_x - log_q_c

                recog_reg_term = tf.reduce_mean(contrib)
                surr_recog_reg_term = recog_reg_term + tf.reduce_mean(
                    tf.stop_gradient(contrib) * self.model.reg_latent_dist.nonreparam_logli(
                        wake_reg_z, wake_reg_z_dist_info
                    )
                )
                recog_reg_term /= self.model.reg_latent_dist.effective_dim
                surr_recog_reg_term /= self.model.reg_latent_dist.effective_dim

                loss += - self.recog_reg_coeff * recog_reg_term
                surr_loss += - self.recog_reg_coeff * surr_recog_reg_term

                tf.scalar_summary("MI_recog", recog_reg_term)
                self.log_vars.append(("MI_recog", recog_reg_term))

            tf.scalar_summary("loss", loss)
            tf.scalar_summary("surr_loss", surr_loss)
            self.log_vars.append(("loss", loss))

            if optimizer:
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.trainer = pt.apply_optimizer(optimizer, losses=[surr_loss])

        with pt.defaults_scope(phase=pt.Phase.test):
            if self.use_info_reg:
                with tf.variable_scope("model", reuse=True) as scope:
                    # we need 10 different codes
                    z_var, _ = self.model.encode(input_tensor)
                    nonreg_z_var = self.model.nonreg_z(z_var)
                    noncat_var = tf.concat(0, [
                        tf.tile(nonreg_z_var[:10, :], [10, 1]),
                        tf.tile(nonreg_z_var[:1, :], [self.batch_size - 100, 1]),
                    ])
                    lookup = np.eye(10, dtype=np.float32)
                    cat_ids = []
                    for idx in xrange(10):
                        cat_ids.extend([idx] * 10)
                    cat_ids.extend([0] * (self.batch_size - 100))
                    cat_var = tf.constant(lookup[np.array(cat_ids)])

                    z_var = tf.concat(1, [noncat_var, cat_var])
                    _, x_dist_info = self.model.decode(z_var)

                    # just take the mean image
                    if isinstance(self.model.output_dist, Bernoulli):
                        img_var = x_dist_info["p"]
                    elif isinstance(self.model.output_dist, Gaussian):
                        img_var = x_dist_info["mean"]
                    else:
                        raise NotImplementedError
                    rows = 10  # int(np.sqrt(FLAGS.batch_size))
                    img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                    img_var = img_var[:rows * rows, :, :, :]
                    imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                    stacked_img = []
                    for row in xrange(rows):
                        row_img = []
                        for col in xrange(rows):
                            row_img.append(imgs[row, col, :, :, :])
                        stacked_img.append(tf.concat(1, row_img))
                    imgs = tf.concat(0, stacked_img)
                imgs = tf.expand_dims(imgs, 0)
                tf.image_summary("image", imgs, max_images=3)
            else:
                with tf.variable_scope("model", reuse=True) as scope:
                    z_var, _ = self.model.encode(input_tensor)
                    _, x_dist_info = self.model.decode(z_var)

                    # just take the mean image
                    if isinstance(self.model.output_dist, Bernoulli):
                        img_var = x_dist_info["p"]
                    elif isinstance(self.model.output_dist, Gaussian):
                        img_var = x_dist_info["mean"]
                    else:
                        raise NotImplementedError
                    rows = 10  # int(np.sqrt(FLAGS.batch_size))
                    img_var = tf.concat(1, [input_tensor, img_var])
                    img_var = tf.reshape(img_var, [self.batch_size*2] + list(self.dataset.image_shape))
                    img_var = img_var[:rows * rows, :, :, :]
                    imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                    stacked_img = []
                    for row in xrange(rows):
                        row_img = []
                        for col in xrange(rows):
                            row_img.append(imgs[row, col, :, :, :])
                        stacked_img.append(tf.concat(1, row_img))
                    imgs = tf.concat(0, stacked_img)
                imgs = tf.expand_dims(imgs, 0)
                tf.image_summary("qz_image", imgs, max_images=3)


                if isinstance(self.model.latent_dist, Mixture):
                    modes = self.model.latent_dist.mode(z_var, self.model.latent_dist.prior_dist_info(self.batch_size))
                    tf.histogram_summary("qz_modes", modes)
                    for i, dist in enumerate(self.model.latent_dist.dists):
                        with tf.variable_scope("model", reuse=True) as scope:
                            z_var = dist.sample_prior(self.batch_size)
                            self.sample_zs.append(z_var)
                            _, x_dist_info = self.model.decode(z_var)

                            # just take the mean image
                            if isinstance(self.model.output_dist, Bernoulli):
                                img_var = x_dist_info["p"]
                            elif isinstance(self.model.output_dist, Gaussian):
                                img_var = x_dist_info["mean"]
                            else:
                                raise NotImplementedError
                            self.sample_imgs.append(img_var)

                            rows = 10  # int(np.sqrt(FLAGS.batch_size))
                            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                            img_var = img_var[:rows * rows, :, :, :]
                            imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                            stacked_img = []
                            for row in xrange(rows):
                                row_img = []
                                for col in xrange(rows):
                                    row_img.append(imgs[row, col, :, :, :])
                                stacked_img.append(tf.concat(1, row_img))
                            imgs = tf.concat(0, stacked_img)
                        imgs = tf.expand_dims(imgs, 0)
                        tf.image_summary("pz_mode%s_image" % i, imgs, max_images=3)
                else:
                    with tf.variable_scope("model", reuse=True) as scope:
                        z_var = self.model.latent_dist.sample_prior(self.batch_size)
                        _, x_dist_info = self.model.decode(z_var)

                        # just take the mean image
                        if isinstance(self.model.output_dist, Bernoulli):
                            img_var = x_dist_info["p"]
                        elif isinstance(self.model.output_dist, Gaussian):
                            img_var = x_dist_info["mean"]
                        else:
                            raise NotImplementedError
                        rows = 10  # int(np.sqrt(FLAGS.batch_size))
                        img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                        img_var = img_var[:rows * rows, :, :, :]
                        imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                        stacked_img = []
                        for row in xrange(rows):
                            row_img = []
                            for col in xrange(rows):
                                row_img.append(imgs[row, col, :, :, :])
                            stacked_img.append(tf.concat(1, row_img))
                        imgs = tf.concat(0, stacked_img)
                    imgs = tf.expand_dims(imgs, 0)
                    tf.image_summary("pz_image", imgs, max_images=3)

    def train(self):

        self.init_opt()


        # with tf.Session() as sess:
        with self.sess.as_default():
            sess = self.sess
            # check = tf.add_check_numerics_ops()
            init = tf.initialize_all_variables()
            sess.run(init)
            if self.bnn_decoder:
                sess.run([
                    self.saved_prior_mean.assign(self.mean_var),
                    self.saved_prior_std.assign(self.std_var),
                ])

            summary_op = tf.merge_all_summaries()
            summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            counter = 0

            log_dict = dict(self.log_vars)
            log_keys = log_dict.keys()
            log_vars = log_dict.values()

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):

                    pbar.update(i)
                    x, _ = self.dataset.train.next_batch(self.true_batch_size)
                    x = np.tile(x, [self.weight_redundancy, 1])

                    if counter == 0:
                        log_vals = sess.run(log_vars, {self.input_tensor: x})
                        log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, log_vals))
                        print("Initial: " + log_line)
                        # import ipdb; ipdb.set_trace()
                    log_vals = sess.run([self.trainer] + log_vars, {self.input_tensor: x})[1:]
                    all_log_vals.append(log_vals)

                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                    if counter % self.summary_interval == 0:
                        summary = tf.Summary()
                        test_x, _ = self.dataset.test.next_batch(self.true_batch_size)
                        test_x = np.tile(test_x, [self.weight_redundancy, 1])
                        test_log_vals = sess.run(log_vars, {self.input_tensor: test_x})
                        for k,v in zip(log_keys, test_log_vals):
                            summary.value.add(
                                tag="test_%s"%k,
                                simple_value=float(v),
                            )
                        summary_str = sess.run(summary_op, {self.input_tensor: x})
                        summary.MergeFromString(summary_str)
                        summary_writer.add_summary(summary, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))

                print(log_line)
                # d1 = (sess.run(self.model.latent_dist.dists[0].prior_dist_info(1)))['mean']
                # d2 = (sess.run(self.model.latent_dist.dists[1].prior_dist_info(1)))['mean']
                # print(np.linalg.norm(d1 - d2), d1, d2)
                # d3 = (sess.run(self.model.latent_dist.dists[2].prior_dist_info(1)))['mean']
                # print(np.linalg.norm(d1 - d3))
                # # sys.stdout.flush()
                # if (epoch+1) % 50 == 0:
                #     import matplotlib.pyplot as plt
                #     plt.close('all')
                #     for d in self.model.latent_dist.dists: plt.figure(); plt.imshow(sess.run(self.model.decode(d.prior_dist_info(1)["mean"])[1])['p'].reshape((28, 28)),cmap='Greys_r'); plt.show(block=False)
                #     import ipdb; ipdb.set_trace()

                # if epoch == 15:
                #     import ipdb; ipdb.set_trace()


    def restore(self):

        self.init_opt(optimizer=False)

        init = tf.initialize_all_variables()

        sess = tf.InteractiveSession()
        saver = tf.train.Saver()
        fn = tf.train.latest_checkpoint(self.checkpoint_dir)
        print(fn)
        saver.restore(sess, fn)
        return sess

    def ipdb(self):
        sess = self.restore()
        import matplotlib.pyplot as plt
        for d in self.model.latent_dist.dists: plt.figure(); plt.imshow(sess.run(self.model.decode(d.prior_dist_info(1)["mean"])[1])['p'].reshape((28, 28)),cmap='Greys_r'); plt.show(block=False)
        import ipdb; ipdb.set_trace()
