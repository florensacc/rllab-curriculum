from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Bernoulli, Gaussian, Mixture
import rllab.misc.logger as logger
import sys
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, logsumexp


class VAE(object):
    def __init__(self,
                 model,
                 dataset,
                 batch_size,
                 exp_name="experiment",
                 optimizer_cls=AdamaxOptimizer,
                 optimizer_args={},
                 # log_dir="logs",
                 # checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=10000,
                 vali_eval_interval=400,
                 # learning_rate=1e-3,
                 summary_interval=100,
                 monte_carlo_kl=False,
                 min_kl=0.,
                 use_prior_reg=False,
                 weight_redundancy=1,
                 bnn_decoder=False,
                 bnn_kl_coeff=1.,
                 k=1, # importance sampling ratio
                 cond_px_ent=None,
                 anneal_after=None,
    ):
        """
        :type model: RegularizedHelmholtzMachine
        :type use_info_reg: bool
        :type info_reg_coeff: float
        :type use_recog_reg: bool
        :type recog_reg_coeff: float
        :type learning_rate: float
        """
        self.optimizer_cls = optimizer_cls
        self.optimizer_args = optimizer_args
        self.anneal_after = anneal_after
        self.cond_px_ent = cond_px_ent
        self.vali_eval_interval = vali_eval_interval
        self.sample_zs = []
        self.sample_imgs = []
        self.model = model
        self.dataset = dataset
        self.true_batch_size = batch_size
        self.batch_size = batch_size * weight_redundancy
        self.weight_redundancy = weight_redundancy
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = logger.get_snapshot_dir()
        self.checkpoint_dir = logger.get_snapshot_dir()
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.summary_interval = summary_interval
        # self.learning_rate = learning_rate
        self.trainer = None
        self.input_tensor = None
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
        self.k = k
        self.eval_batch_size = self.batch_size / k
        self.eval_input_tensor = None
        self.eval_log_vars = []

    def init_opt(self, init=False, eval=False):
        if init:
            self.model.init_mode()
        else:
            self.model.train_mode()

        log_vars = []
        if eval:
            self.eval_input_tensor = \
                input_tensor = \
                tf.placeholder(
                    tf.float32,
                    [self.eval_batch_size, self.dataset.image_dim],
                    "eval_input"
                )
        else:
            self.input_tensor = input_tensor = tf.placeholder(
                tf.float32,
                [self.batch_size, self.dataset.image_dim],
                "train_input_init_%s" % init
            )

        with pt.defaults_scope(phase=pt.Phase.train):
            z_var, log_p_z_given_x, z_dist_info = \
                self.model.encode(input_tensor, k=self.k if eval else 1)
            x_var, x_dist_info = self.model.decode(z_var)

            log_p_x_given_z = self.model.output_dist.logli(
                tf.reshape(
                    tf.tile(input_tensor, [1, self.k]),
                    [-1, self.dataset.image_dim],
                ) if eval else input_tensor,
                x_dist_info
            )

            ndim = self.model.output_dist.effective_dim
            if eval:
                assert self.monte_carlo_kl
                kls = (
                        - self.model.latent_dist.logli_prior(z_var) \
                        + log_p_z_given_x
                    )
                kl = tf.reduce_mean(kls)


                true_vlb = tf.reduce_mean(
                    logsumexp(tf.reshape(log_p_x_given_z - kls, [-1, self.k])),
                ) - np.log(self.k)
                # this is shaky but ok since we are just in eval mode
                vlb = tf.reduce_mean(log_p_x_given_z) - \
                      tf.maximum(kl, self.min_kl * ndim)
                log_vars.append((
                    "true_vlb_sum_k0",
                    tf.reduce_mean(log_p_x_given_z - kls)
                ))
            else:
                if self.monte_carlo_kl:
                    # Construct the variational lower bound
                    kl = tf.reduce_mean(
                        - self.model.latent_dist.logli_prior(z_var)  \
                        + log_p_z_given_x
                    )
                else:
                    # Construct the variational lower bound
                    kl = tf.reduce_mean(self.model.latent_dist.kl_prior(z_dist_info))

                true_vlb = tf.reduce_mean(log_p_x_given_z) - kl
                vlb = tf.reduce_mean(log_p_x_given_z) - tf.maximum(kl, self.min_kl * ndim)

            # ld = self.model.latent_dist
            # prior_z_var = ld.sample_prior(self.batch_size)
            # prior_reg = tf.reduce_mean(
            #     ld.logli_prior(prior_z_var) - \
            #         ld.logli_init_prior(prior_z_var)
            # )
            # emp_prior_reg = tf.reduce_mean(
            #     ld.logli_prior(z_var) - \
            #     ld.logli_init_prior(z_var)
            # )
            # self.log_vars.append(("prior_reg", prior_reg))
            # self.log_vars.append(("emp_prior_reg", emp_prior_reg))
            # if self.use_prior_reg:
            #     vlb -= prior_reg


            # surr_vlb = vlb + tf.reduce_mean(
            #         tf.stop_gradient(log_p_x_given_z) * self.model.latent_dist.nonreparam_logli(z_var, z_dist_info)
            #     )
            # surr_vlb
            # Normalize by the dimensionality of the data distribution
            log_vars.append(("vlb_sum", vlb))
            log_vars.append(("kl_sum", kl))
            log_vars.append(("true_vlb_sum", true_vlb))

            true_vlb /= ndim
            vlb /= ndim
            # surr_vlb /= ndim
            kl /= ndim

            # loss = - vlb
            # surr_loss = - surr_vlb

            log_vars.append((
                "ent_x_given_z",
                tf.reduce_mean(self.model.output_dist.entropy(x_dist_info)) / ndim
            ))
            log_vars.append(("vlb", vlb))
            log_vars.append(("kl", kl))
            log_vars.append(("true_vlb", true_vlb))
            final_losses = [-vlb]

            if self.cond_px_ent:
                ld = self.model.latent_dist
                prior_z_var = ld.sample_prior(self.batch_size)
                _, prior_x_dist_info = self.model.decode(prior_z_var)
                prior_ent = tf.reduce_mean(self.model.output_dist.entropy(prior_x_dist_info)) \
                            / ndim
                # loss += self.cond_px_ent * prior_ent
                final_losses.append(self.cond_px_ent * prior_ent)
                log_vars.append((
                    "prior_ent_x_given_z",
                    prior_ent
                ))
            self.init_hook(locals())

            if (not init) and (not eval):
                for name, var in self.log_vars:
                    tf.scalar_summary(name, var)
                with tf.variable_scope("optim"):
                    # optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    optimizer = self.optimizer_cls # AdamaxOptimizer(self.learning_rate)
                    if self.anneal_after is not None:
                        self.lr_var = tf.Variable(
                            # assume lr is always set
                            initial_value=self.optimizer_args["learning_rate"],
                            name="opt_lr",
                        )
                        self.optimizer_args["learning_rate"] = self.lr_var
                    optimizer = self.optimizer_cls(**self.optimizer_args)
                    self.trainer = pt.apply_optimizer(optimizer, losses=final_losses)

        if init:
            # destroy all summaries
            tf.get_collection_ref(tf.GraphKeys.SUMMARIES)[:] = []
            # no pic summary
            return
        if eval:
            self.eval_log_vars = log_vars
            tf.get_collection_ref(tf.GraphKeys.SUMMARIES)[:] = []
            return

        self.log_vars = log_vars

        with pt.defaults_scope(phase=pt.Phase.test):
                with tf.variable_scope("model", reuse=True) as scope:
                    # z_var, _ = self.model.encode(input_tensor)
                    # _, x_dist_info = self.model.decode(z_var)

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

    def prepare_feed(self, data, bs):
        x, _ = data.next_batch(bs)
        x = np.tile(x, [self.weight_redundancy, 1])
        return {
            self.input_tensor: x,
        }

    def prepare_eval_feed(self, data, bs):
        x, _ = data.next_batch(bs)
        x = np.tile(x, [self.weight_redundancy, 1])
        return {
            self.eval_input_tensor: x,
        }

    def train(self):
        sess = tf.Session()
        self.sess = sess

        self.init_opt(init=True)

        with self.sess.as_default():
            sess = self.sess
            # check = tf.add_check_numerics_ops()
            init = tf.initialize_all_variables()
            if self.bnn_decoder:
                assert False

            saver = tf.train.Saver()

            counter = 0

            for epoch in range(self.max_epoch):
                logger.log("epoch %s" % epoch)
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):

                    pbar.update(i)
                    # x, _ = self.dataset.train.next_batch(self.true_batch_size)
                    # x = np.tile(x, [self.weight_redundancy, 1])
                    feed = self.prepare_feed(self.dataset.train, self.true_batch_size)

                    if counter == 0:
                        sess.run(init, feed)
                        self.init_opt(init=False, eval=True)
                        self.init_opt(init=False, eval=False)
                        vs = tf.all_variables()
                        sess.run(tf.initialize_variables([
                            v for v in vs if "optim" in v.name or "global_step" in v.name
                        ]))
                        print("vars initd")

                        log_dict = dict(self.log_vars)
                        log_keys = log_dict.keys()
                        log_vars = log_dict.values()
                        eval_log_dict = dict(self.eval_log_vars)
                        eval_log_keys = eval_log_dict.keys()
                        eval_log_vars = eval_log_dict.values()

                        summary_op = tf.merge_all_summaries()
                        summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                        feed = self.prepare_feed(self.dataset.train, self.true_batch_size)
                        log_vals = sess.run([] + log_vars, feed)[:]
                        log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, log_vals))
                        print("Initial: " + log_line)

                    log_vals = sess.run(
                        [self.trainer] + log_vars,
                        feed
                    )[1:]
                    all_log_vals.append(log_vals)


                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                    if counter % self.summary_interval == 0:
                        summary = tf.Summary()
                        summary_str = sess.run(summary_op, feed)
                        summary.MergeFromString(summary_str)
                        if counter % self.vali_eval_interval == 0:
                            ds = self.dataset.validation
                            all_test_log_vals = []
                            for ti in xrange(ds.images.shape[0] / self.eval_batch_size):
                                # test_x, _ = self.dataset.validation.next_batch(self.eval_batch_size)
                                # test_x = np.tile(test_x, [self.weight_redundancy, 1])
                                eval_feed = self.prepare_eval_feed(
                                    self.dataset.validation,
                                    self.eval_batch_size,
                                )
                                test_log_vals = sess.run(
                                    eval_log_vars,
                                    eval_feed,
                                )
                                all_test_log_vals.append(test_log_vals)
                            avg_test_log_vals = np.mean(np.array(all_test_log_vals), axis=0)
                            log_line = "EVAL" + "; ".join("%s: %s" % (str(k), str(v))
                                                      for k, v in zip(eval_log_keys, avg_test_log_vals))
                            logger.log(log_line)
                            for k,v in zip(log_keys, avg_test_log_vals):
                                summary.value.add(
                                    tag="vali_%s"%k,
                                    simple_value=float(v),
                                )
                        summary_writer.add_summary(summary, counter)
                    # need to ensure avg_test_log is always available
                    counter += 1

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))

                logger.log(log_line)
                for k,v in zip(log_keys, avg_log_vals):
                    logger.record_tabular("train_%s"%k, v)
                for k,v in zip(eval_log_keys, avg_test_log_vals):
                    logger.record_tabular("vali_%s"%k, v)
                logger.dump_tabular(with_prefix=False)

                if epoch >= self.anneal_after:
                    if (epoch % 100) == 0:
                        lr_val = sess.run([
                            self.lr_var.assign(
                                self.lr_var * 0.75
                            )
                        ])
                        logger.log("Learning rate annealed to %s" % lr_val)


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

    def init_hook(self, vars):
        pass
