from collections import defaultdict

from pip._vendor.distlib.locators import locate

from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Bernoulli, Gaussian, Mixture, DiscretizedLogistic, ConvAR
import rllab.misc.logger as logger
import sys
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, logsumexp, flatten, assign_to_gpu, \
    average_grads, temp_restore


# model sharing vae
class ShareVAE(object):
    def __init__(
            self,
            model,
            dataset,
            batch_size,
            exp_name="experiment",
            optimizer_cls=AdamaxOptimizer,
            optimizer_args={},
            # log_dir="logs",
            checkpoint_dir=None,
            resume_from=None,
            max_epoch=100,
            updates_per_epoch=None,
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
            anneal_every=100,
            anneal_factor=0.75,
            exp_avg=None,
            l2_reg=None,
            img_on=True,
            kl_coeff=1.,
            kl_coeff_spec=None,
            noise=True,
            vis_ar=True,
            num_gpus=1,
            min_kl_onesided=False, # True if prior doesn't see freebits
            slow_kl=False,
            lwarm_until=None,
            arwarm_until=None,
            staged=False,
            unconditional=False,
            resume_includes=None,
    ):
        """
        :type model: RegularizedHelmholtzMachine
        :type use_info_reg: bool
        :type info_reg_coeff: float
        :type use_recog_reg: bool
        :type recog_reg_coeff: float
        :type learning_rate: float

        Parameters
        ----------
        """
        self.resume_includes = resume_includes
        self.unconditional = unconditional
        self.staged = staged
        self.arwarm_until = arwarm_until
        self.lwarm_until = lwarm_until
        self.slow_kl = slow_kl
        self.min_kl_onesided = min_kl_onesided
        self.num_gpus = num_gpus
        self._vis_ar = vis_ar
        self.resume_from = resume_from
        self.checkpoint_dir = checkpoint_dir or logger.get_snapshot_dir()
        if isinstance(optimizer_cls, str):
            optimizer_cls = eval(optimizer_cls)
        self.noise = noise
        self.kl_coeff_spec = kl_coeff_spec
        if kl_coeff_spec is None:
            self.kl_coeff = kl_coeff
        else:
            self.kl_coeff = tf.Variable(
                initial_value=kl_coeff_spec.start,
                trainable=False,
                name="kl_coeff"
            )
        self.anneal_factor = anneal_factor
        self.anneal_every = anneal_every
        self.img_on = img_on
        self.l2_reg = l2_reg
        self.exp_avg = exp_avg
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
        self.exp_name = exp_name[:20] # tf doesnt like filenames that are too long
        self.log_dir = logger.get_snapshot_dir()
        self.snapshot_interval = snapshot_interval
        if updates_per_epoch:
            input("should not set updates_per_epoch")
            self.updates_per_epoch = updates_per_epoch
        else:
            self.updates_per_epoch = dataset.train.images.shape[0] // batch_size
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
        self.eval_batch_size = self.batch_size // k
        self.eval_input_tensor = None
        self.eval_log_vars = []
        self.sym_vars = {}
        self.ema = None

        bs_per_gpu = self.batch_size // self.num_gpus
        self.x_sample_holder = tf.placeholder(tf.float32, shape=(bs_per_gpu, 32, 32, 3))

        if self.slow_kl:
            self.ema_kl = tf.Variable(initial_value=0., trainable=False, name="ema_kl")
        if self.staged:
            # for data-init, this should be set to 1.
            self.staged_cond_mask = tf.Variable(initial_value=1., trainable=False, name="cond_mask")
        else:
            self.staged_cond_mask = 1.

        assert not self.cond_px_ent
        assert not self.l2_reg

        with tf.variable_scope("optim"):
            # optimizer = tf.train.AdamOptimizer(self.learning_rate)
            if self.anneal_after is not None:
                self.lr_var = tf.Variable(
                    # assume lr is always set
                    initial_value=self.optimizer_args["learning_rate"],
                    name="opt_lr",
                    trainable=False,
                )
                self.optimizer_args["learning_rate"] = self.lr_var
            self.optimizer = self.optimizer_cls(**self.optimizer_args)

    def init_opt(self, init=False, eval=False, opt_off=False):
        if init:
            self.model.init_mode()
        else:
            self.model.train_mode(eval=eval)

        # log_vars = []
        dict_log_vars = defaultdict(list)
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


        grads = []
        xs = tf.split(0, self.num_gpus, input_tensor)

        pixelcnn = self.model.output_dist

        for i in range(
            1 if init else self.num_gpus
        ):
            x = xs[i]
            with tf.device(assign_to_gpu(i)):
                # get activations of pixelcnn
                causal_feats = pixelcnn.infer_temp(
                    x
                )

                z_var, log_p_z_given_x, z_dist_info = \
                    self.model.encode(causal_feats, k=self.k if eval else 1)

                if not self.noise:
                    z_var = z_dist_info["mean"]
                    log_p_x_given_z = 1.

                # get raw conditional feats to avoid changing helmhpltz machine
                cond_feats = self.model.decode(z_var, raw=True)
                x_dist_info = dict(
                    causal_feats=causal_feats,
                    cond_feats=cond_feats * self.staged_cond_mask,
                )
                if self.unconditional:
                    x_dist_info["cond_feats"] = 0. * x_dist_info["cond_feats"]

                log_p_x_given_z = self.model.output_dist.logli(
                    tf.reshape(
                        tf.tile(x, [1, self.k]),
                        [-1, self.dataset.image_dim],
                    ) if eval else x,
                    x_dist_info
                )

                ndim = self.model.output_dist.effective_dim
                log_p_z = self.model.latent_dist.logli_prior(z_var)
                dict_log_vars["log_p_z"].append(
                    tf.reduce_mean(log_p_z)
                )

                if eval:
                    assert self.monte_carlo_kl
                    kls = (
                            - log_p_z \
                            + log_p_z_given_x
                        )
                    kl = tf.reduce_mean(kls)

                    true_vlb = tf.reduce_mean(
                        logsumexp(tf.reshape(
                            log_p_x_given_z - (kls if self.kl_coeff != 0. else 0.),
                            [-1, self.k])),
                    ) - np.log(self.k)
                    # this is shaky but ok since we are just in eval mode
                    vlb = tf.reduce_mean(log_p_x_given_z) - \
                          (
                              tf.maximum(kl, self.min_kl * ndim) * self.kl_coeff
                              if self.kl_coeff != 0. else 0.
                          )
                    dict_log_vars["true_vlb_sum_k0"].append(
                        tf.reduce_mean(log_p_x_given_z - kls)
                    )
                else:
                    if self.monte_carlo_kl:
                        # Construct the variational lower bound
                        kl = tf.reduce_mean(
                            - log_p_z  \
                            + log_p_z_given_x
                        )
                    else:
                        # Construct the variational lower bound
                        kl = tf.reduce_mean(self.model.latent_dist.kl_prior(z_dist_info))

                    true_vlb = tf.reduce_mean(log_p_x_given_z) - (
                        kl if self.kl_coeff != 0. else 0.
                    )
                    if self.min_kl_onesided:
                        avg_log_p_sg_z = tf.reduce_mean(
                            # self.model.latent_dist.logli_prior(
                            #     tf.stop_gradient(z_var)
                            # )
                            log_p_z # this variant lets q(z|x) seek high density region
                                    # but not pay the price
                        )

                        # when freebits is enabled, still give gradients to prior
                        # but maintain the original numerical values
                        vlb = tf.reduce_mean(log_p_x_given_z) - \
                              tf.maximum(
                                  kl,
                                  self.min_kl * ndim
                                  - avg_log_p_sg_z
                                  + tf.stop_gradient(avg_log_p_sg_z)
                              ) * self.kl_coeff
                    else:
                        if self.slow_kl:
                            ema_kl = self.ema_kl
                            dict_log_vars["ema_kl"].append(ema_kl / ndim)
                            ema_kl_decay = 0.99
                            if self.slow_kl is True:
                                # "soft" version
                                kl, _ = tf.tuple([
                                    kl,
                                    ema_kl.assign(ema_kl*ema_kl_decay+ kl*(1.-ema_kl_decay))
                                ])
                            elif self.slow_kl == "hard":
                                pass
                                # rely on training algo
                            else:
                                raise ValueError()
                            surr_kl = tf.select(
                                ema_kl >= self.min_kl*ndim,
                                kl,
                                self.min_kl*ndim
                            )
                        else:
                            surr_kl = tf.maximum(
                                kl,
                                self.min_kl * ndim
                            )
                        vlb = tf.reduce_mean(log_p_x_given_z) - (
                            surr_kl * self.kl_coeff * self.staged_cond_mask
                            if self.kl_coeff != 0 else 0.
                        )

                # Normalize by the dimensionality of the data distribution
                dict_log_vars["vlb_sum"].append(vlb)
                dict_log_vars["kl_sum"].append(kl)
                dict_log_vars["true_vlb_sum"].append(true_vlb)
                dict_log_vars["cond_logp"].append(true_vlb + kl)

                true_vlb /= ndim
                vlb /= ndim
                # surr_vlb /= ndim
                kl /= ndim

                dict_log_vars["vlb"].append(vlb)
                dict_log_vars["kl"].append(kl)
                dict_log_vars["true_vlb"].append(true_vlb)
                dict_log_vars["bits/dim"].append(true_vlb/np.log(2.))
                # final_losses = [-vlb]

                surr_loss = -vlb
                tower_grads = None
                self.init_hook(locals())
                if tower_grads is None:
                    tower_grads = self.optimizer.compute_gradients(surr_loss)
                grads.append(tower_grads)

        if init and self.exp_avg is not None:
            self.ema = tf.train.ExponentialMovingAverage(decay=self.exp_avg)
            self.ema_applied = self.ema.apply(tf.trainable_variables())
            self.avg_dict = self.ema.variables_to_restore()

        log_vars = [
            (key, tf.add_n(vals) / self.num_gpus)
            for key, vals in dict_log_vars.items()
        ]
        if (not init) and (not eval) and (not opt_off):
            for name, var in log_vars:
                tf.scalar_summary(name, var)

            with tf.variable_scope("optim"):
                self.trainer = self.optimizer.apply_gradients(
                    grads_and_vars=average_grads(grads)
                )
                if self.exp_avg is not None:
                    with tf.name_scope(None):
                        self.trainer = tf.group(*[self.trainer, self.ema_applied])

        if init:
            # destroy all summaries
            tf.get_collection_ref(tf.GraphKeys.SUMMARIES)[:] = []
            # no pic summary
            return
        # save relevant sym vars
        self.sym_vars[
            "eval" if eval else "train"
        ] = dict(locals())
        if eval:
            self.eval_log_vars = log_vars
            tf.get_collection_ref(tf.GraphKeys.SUMMARIES)[:] = []
            return

        self.log_vars = log_vars

        if not self.img_on:
            return
        with pt.defaults_scope(phase=pt.Phase.test):
                rows = int(np.sqrt(self.true_batch_size))# 10  # int(np.sqrt(FLAGS.batch_size))
                with tf.variable_scope("model", reuse=True) as scope:
                    # z_var, _ = self.model.encode(input_tensor)
                    # _, x_dist_info = self.model.decode(z_var)

                    # just take the mean image
                    if isinstance(self.model.output_dist, Bernoulli):
                        img_var = x_dist_info["p"]
                    elif isinstance(self.model.output_dist, Gaussian):
                        img_var = x_dist_info["mean"]
                    elif isinstance(self.model.output_dist, DiscretizedLogistic):
                        img_var = x_dist_info["mu"]
                    else:
                        img_var = x_var
                        # raise NotImplementedError
                    # rows = 10  # int(np.sqrt(FLAGS.batch_size))
                    img_var = tf.concat(
                        1,
                        list(map(flatten, [input_tensor, img_var]))
                    )
                    img_var = tf.reshape(img_var, [self.batch_size*2] + list(self.dataset.image_shape))
                    img_var = img_var[:rows * rows, :, :, :]
                    imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                    stacked_img = []
                    for row in range(rows):
                        row_img = []
                        for col in range(rows):
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

                            # rows = 10  # int(np.sqrt(FLAGS.batch_size))
                            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                            img_var = img_var[:rows * rows, :, :, :]
                            imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                            stacked_img = []
                            for row in range(rows):
                                row_img = []
                                for col in range(rows):
                                    row_img.append(imgs[row, col, :, :, :])
                                stacked_img.append(tf.concat(1, row_img))
                            imgs = tf.concat(0, stacked_img)
                        imgs = tf.expand_dims(imgs, 0)
                        tf.image_summary("pz_mode%s_image" % i, imgs, max_images=3)
                else:
                    with tf.variable_scope("model", reuse=True) as scope:
                        z_var = self.model.latent_dist.sample_prior(self.batch_size)
                        x_var, x_dist_info = self.model.decode(z_var)

                        # just take the mean image
                        if isinstance(self.model.output_dist, Bernoulli):
                            img_var = x_dist_info["p"]
                        elif isinstance(self.model.output_dist, Gaussian):
                            img_var = x_dist_info["mean"]
                        else:
                            img_var = x_var
                        # rows = int(np.sqrt(self.true_batch_size))# 10  # int(np.sqrt(FLAGS.batch_size))
                        img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
                        img_var = img_var[:rows * rows, :, :, :]
                        imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
                        stacked_img = []
                        for row in range(rows):
                            row_img = []
                            for col in range(rows):
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

        prev_bits = -10.

        with self.sess.as_default():
            sess = self.sess
            init = tf.initialize_all_variables()
            if self.bnn_decoder:
                assert False

            saver = tf.train.Saver()

            counter = 0
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                # print(shape) print(len(shape))
                variable_parametes = 1
                for dim in shape:
                    # print(dim)
                    variable_parametes *= dim.value
                # print(variable_parametes)
                total_parameters += variable_parametes
            logger.log("# of parameter vars %s" % len(tf.trainable_variables()))
            logger.log("total parameters %s" % total_parameters)

            keys = ["cond", "infer", ]
            for key in keys:
                counter = 0
                total_parameters = 0
                for variable in tf.trainable_variables():
                    if key not in variable.name:
                        continue
                    # shape is an array of tf.Dimension
                    shape = variable.get_shape()
                    # print(shape) print(len(shape))
                    variable_parametes = 1
                    for dim in shape:
                        # print(dim)
                        variable_parametes *= dim.value
                    # print(variable_parametes)
                    total_parameters += variable_parametes
                logger.log("# of parameter vars %s" % len(tf.trainable_variables()))
                logger.log("%s total parameters %s" % (key,total_parameters))

            for epoch in range(self.max_epoch):
                self.pre_epoch(epoch, locals())

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
                        if self.resume_from is None or (self.resume_includes is not None):
                            sess.run(init, feed)
                        self.init_opt(init=False, eval=True)
                        self.init_opt(init=False, eval=False)
                        vs = tf.all_variables()
                        sess.run(tf.initialize_variables([
                            v for v in vs if
                                "optim" in v.name or "global_step" in v.name or \
                                ("cv_coeff" in v.name)
                        ]))
                        print("vars initd")
                        if self.staged:
                            sess.run(
                                self.staged_cond_mask.assign(0.)
                            )

                        if self.resume_from is not None:
                            # print("not resuming")
                            print("resuming from %s" % self.resume_from)
                            fn = tf.train.latest_checkpoint(self.resume_from)
                            print("latest ckpt: %s" % fn)
                            if self.resume_includes is None:
                                saver.restore(sess, fn)
                            else:
                                vars_to_resume = [
                                    v for v in vs if any([
                                        include in v.name for include in self.resume_includes
                                    ]) and "optim" not in v.name
                                ]
                                print("resuming", [v.name for v in vars_to_resume])
                                resume_saver = tf.train.Saver(var_list=vars_to_resume)
                                resume_saver.restore(sess, fn)
                            print("resumed")

                        log_dict = dict(self.log_vars)
                        log_keys = list(log_dict.keys())
                        log_vars = list(log_dict.values())
                        eval_log_dict = dict(self.eval_log_vars)
                        eval_log_keys = list(eval_log_dict.keys())
                        eval_log_vars = list(eval_log_dict.values())

                        summary_op = tf.merge_all_summaries()
                        summary_writer = tf.train.SummaryWriter(self.log_dir, sess.graph)

                        feed = self.prepare_feed(self.dataset.train, self.true_batch_size)

                        if self.resume_from:
                            self.ar_vis(sess, feed, locals())

                            # print("resumption ema eval")
                            # with temp_restore(sess, self.ema):
                            #     ds = self.dataset.validation
                            #     all_test_log_vals = []
                            #     for ti in range(ds.images.shape[0] // self.eval_batch_size):
                            #         # test_x, _ = self.dataset.validation.next_batch(self.eval_batch_size)
                            #         # test_x = np.tile(test_x, [self.weight_redundancy, 1])
                            #         eval_feed = self.prepare_eval_feed(
                            #             self.dataset.validation,
                            #             self.eval_batch_size,
                            #         )
                            #         test_log_vals = sess.run(
                            #             eval_log_vars,
                            #             eval_feed,
                            #         )
                            #         all_test_log_vals.append(test_log_vals)
                            #
                            # avg_test_log_vals = np.mean(np.array(all_test_log_vals), axis=0)
                            # log_line = "EVAL" + "; ".join("%s: %s" % (str(k), str(v))
                            #                               for k, v in zip(eval_log_keys, avg_test_log_vals))
                            # print(log_line)

                        log_vals = sess.run([] + log_vars, feed)[:]
                        log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, log_vals))
                        print(("Initial: " + log_line))
                        # import ipdb; ipdb.set_trace()

                    # go = dict(locals())
                    # del go["self"]
                    # self.iter_hook(**go)
                    self.iter_hook(
                       sess=sess, counter=counter, feed=feed
                    )

                    # if i == 20:
                    #     run_metadata = tf.RunMetadata()
                    #     log_vals = sess.run(
                    #         [self.trainer] + log_vars,
                    #         feed,
                    #         options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                    #         run_metadata=run_metadata,
                    #     )[1:]
                    #     from tensorflow.python.client import timeline
                    #     trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                    #     trace_file = open('timeline_noexp_gpups.ctf.json', 'w')
                    #     trace_file.write(trace.generate_chrome_trace_format())
                    #     trace_file.close()
                    #     import ipdb; ipdb.set_trace()

                    log_vals = sess.run(
                        [self.trainer] + log_vars,
                        feed
                    )[1:]
                    all_log_vals.append(log_vals)
                    # if any(np.any(np.isnan(val)) for val in log_vals):
                    #     print("NaN detected! ")
                    #     # if self.ema:
                    #     #     print("restoring from ema")
                    #     #     ema_out = sess.run(
                    #     #         [tf.assign(var, avg) for var, avg in self.ema._averages.items()]
                    #     #     )
                    #     # else:
                    #     print("aborting")
                    #     exit(1)


                    if counter != 0 and counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print(("Model saved in file: %s" % fn))

                    if counter % self.summary_interval == 0:
                        summary = tf.Summary()
                        try:
                            summary_str = sess.run(summary_op, feed)
                        except tf.python.framework.errors.InvalidArgumentError as e:
                            # sometimes there are transient errors
                            print("Ignoring %s"%e)
                        summary.MergeFromString(summary_str)
                        if counter % self.vali_eval_interval == 0:
                            # hijack summay interval to do ar sampling
                            if isinstance(self.model.output_dist, ConvAR):
                                if counter != 0:
                                    self.ar_vis(sess, feed, locals())

                            with temp_restore(sess, self.ema):
                                ds = self.dataset.validation
                                all_test_log_vals = []
                                for ti in range(ds.images.shape[0] // self.eval_batch_size):
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
                                    # fast eval for the first itr
                                    if counter == 0 and (self.resume_from is None):
                                        if ti >= 4:
                                            break

                            avg_test_log_vals = np.mean(np.array(all_test_log_vals), axis=0)
                            log_line = "EVAL" + "; ".join("%s: %s" % (str(k), str(v))
                                                      for k, v in zip(eval_log_keys, avg_test_log_vals))
                            logger.log(log_line)
                            for k, v in zip(log_keys, avg_test_log_vals):
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
                for lk, lks, lraw in [
                    ("train", log_keys, all_log_vals),
                    ("vali", eval_log_keys, all_test_log_vals),
                ]:
                    for k, v in zip(
                            lks,
                            # avg_log_vals
                            np.array(lraw).T
                    ):
                        if k == "kl":
                            logger.record_tabular_misc_stat("%s_%s"%(lk,k), v)
                        else:
                            logger.record_tabular("%s_%s"%(lk,k), np.mean(v))

                        if self.slow_kl == "hard" and lk == "train" and k == "kl_sum":
                            sess.run(
                                self.ema_kl.assign(np.mean(v))
                            )

                        if lk == "train":
                            if self.staged:
                                if k == "bits/dim":
                                    cur_bits = np.mean(v)
                                    if epoch != 0:
                                        if cur_bits <= prev_bits + 0.1:
                                            print("cond turned on!! cur: %s, prev: %s" % (cur_bits, prev_bits))
                                            self.staged = False
                                            sess.run(
                                                self.staged_cond_mask.assign(1.)
                                            )
                                    prev_bits = prev_bits*0.9 + cur_bits*0.1

                # for k,v in zip(eval_log_keys, avg_test_log_vals):
                #     logger.record_tabular("vali_%s"%k, v)

                logger.dump_tabular(with_prefix=False)
                print("staged: %s" % self.staged)

                if self.anneal_after is not None and epoch >= self.anneal_after:
                    if (epoch % self.anneal_every) == 0:
                        lr_val = sess.run([
                            self.lr_var.assign(
                                self.lr_var * self.anneal_factor
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

    def pre_epoch(self, epoch, kw):
        if self.lwarm_until is not None:
            if epoch == self.lwarm_until:
                assert self.lwarm_until != 0
                inp_mask = kw["sess"].run(
                    self.model.output_dist._inp_mask.assign(
                        1.
                    )
                )
                print("local ar ON: %s"%inp_mask)
            elif epoch == 0:
                # inp_mask = kw["sess"].run(
                #     self.model.output_dist._inp_mask.assign(
                #         0.
                #     )
                # )
                assert self.model.output_dist._sanity
        if self.arwarm_until is not None:
            assert self.lwarm_until is None
            if epoch == self.arwarm_until:
                assert self.arwarm_until != 0
                inp_mask = kw["sess"].run(
                    self.model.output_dist._context_mask.assign(
                        1.
                    )
                )
                print("code ON: %s"%inp_mask)
            elif epoch == 0:
                assert self.model.output_dist._sanity2
        if self.kl_coeff_spec is not None:
            spec = self.kl_coeff_spec
            if epoch <= spec.end:
                # repeated nodes creations but whatever
                desired = spec.start + (spec.end - spec.start) / spec.length * epoch
                kw["sess"].run(
                    self.kl_coeff.assign(
                        desired
                    )
                )



    def iter_hook(self, **kw):
        pass

    def ar_vis(self, sess, feed, vars):
        import scipy
        if not self._vis_ar:
            return
        dist = self.model.output_dist
        bs_per_gpu = self.batch_size // self.num_gpus
        x_var, context_var, go_sym, tgt_dist = dist.infer_sym(
            bs_per_gpu
        )
        def custom_imsave(name, img):
            assert img.ndim == 3
            if img.shape[2] == 3:
                img = (img + 0.5) * 256
            elif img.shape[2] == 1:
                img = img.reshape(img.shape[:2])
            scipy.misc.imsave(
                name,
                img
            )

        # samples
        fol = "%s/samples/" % self.checkpoint_dir
        try:
            import os
            os.makedirs(fol)
        except:
            pass
        z_var = self.model.latent_dist.sample_prior(bs_per_gpu)
        cond_feats = self.model.decode(z_var, raw=True)
        new_x_gen = self.model.output_dist.sample_one_step(self.x_sample_holder, dict(cond_feats=cond_feats))
        def sample_from_model(sess):
            x_gen = np.zeros((bs_per_gpu,32,32,3), dtype=np.float32)
            for yi in range(32):
                for xi in range(32):
                    new_x_gen_np = sess.run(new_x_gen, {self.x_sample_holder: x_gen})
                    x_gen[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:].copy()
            return x_gen
        import sandbox.pchen.InfoGAN.infogan.misc.imported.plotting as plotting
        sample_x = sample_from_model(sess)
        img_tile = plotting.img_tile(sample_x, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title='CIFAR10 samples')
        plotting.plt.savefig(fol + str(vars["counter"]) + '.png')
        plotting.plt.close('all')

        # # decompress
        # originals = tuple(feed.items())[0][1][-bs_per_gpu:].reshape(batch_imshp)
        # context = sess.run(self.sym_vars["train"]["x_dist_info"], feed)
        # fol = "%s/decomp/" % self.checkpoint_dir
        # try:
        #     import os
        #     os.makedirs(fol)
        # except:
        #     pass
        # # originals = tuple(feed.items())[0][1].reshape([128,32,32,3])
        # # cur_zeros = np.copy(self.dataset.train.images[:128].reshape([-1,32,32,3]))
        # for ix in range(bs_per_gpu):
        #     custom_imsave(
        #         "%s/%s_step%s.png" % (fol, ix, 0),
        #         originals[ix]
        #     )
        # cur_zeros = np.zeros_like(originals)
        # for yi in range(h):
        #     for xi in range(w):
        #         ind = xi + yi * w
        #         if (ind % 8 == 1):
        #             for ix in range(bs_per_gpu):
        #                 custom_imsave(
        #                     "%s/%s_step%s.png" % (fol, ix, ind),
        #                     cur_zeros[ix]
        #                 )
        #         proposal_zeros, tgt_dist_zeros = sess.run([go_sym, tgt_dist], {
        #             x_var: cur_zeros,
        #             context_var: context['context'],
        #         })
        #         cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :].copy()
        # os.system(
        #     "tar -zcvf %s/../decomp.tar.gz %s" % (fol, fol)
        # )
        #
        # if True:
        #     # inpainting
        #     fol = "%s/inpaint/" % self.checkpoint_dir
        #     try:
        #         import os
        #         os.makedirs(fol)
        #     except:
        #         pass
        #
        #     # originals = tuple(feed.items())[0][1].reshape([128,32,32,3])
        #     # cur_zeros = np.copy(self.dataset.train.images[:128].reshape([-1,32,32,3]))
        #     for ix in range(10):
        #         custom_imsave(
        #             "%s/%s_step%s.png" % (fol, ix, 0),
        #             originals[ix]
        #         )
        #
        #     cur_zeros = np.copy(originals)
        #     cur_zeros[:, 16, 16:, :] = 0.
        #     cur_zeros[:, 17:, :, :] = 0.
        #     for yi in range(16, h):
        #         for xi in range(16 if yi == 16 else 0, w):
        #             ind = xi + yi * w
        #             for ix in range(10):
        #                 custom_imsave(
        #                     "%s/%s_step%s.png" % (fol, ix, ind),
        #                     cur_zeros[ix]
        #                 )
        #             proposal_zeros, tgt_dist_zeros = sess.run([go_sym, tgt_dist], {
        #                 x_var: cur_zeros,
        #                 context_var: context['context'],
        #             }
        #                                                       )
        #             cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :].copy()
        #     os.system(
        #         "tar -zcvf %s/../inpaint.tar.gz %s" % (fol, fol)
        #     )
        #
        # if False:
        #     # inspect all leakage
        #     cur_ori = originals
        #     cur_zeros = np.zeros_like(originals)
        #     for yi in range(32):
        #         for xi in range(32):
        #             proposal_zeros, tgt_dist_zeros = sess.run([go_sym, tgt_dist], {
        #                 x_var: cur_zeros,
        #                 context_var: context['context'],
        #             }
        #                                                       )
        #             proposal_ori, tgt_dist_ori = sess.run([go_sym, tgt_dist], {
        #                 x_var: cur_ori,
        #                 context_var: context['context'],
        #             }
        #                                                   )
        #             cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :]
        #             cur_ori[:, yi, xi, :] = proposal_zeros[:, yi, xi, :]
        #             ind = xi + yi * 32
        #             check = np.allclose(
        #                 tgt_dist_zeros['infos'][0][ind],
        #                 tgt_dist_ori['infos'][0][ind],
        #                 atol=2e-5,
        #             )
        #             print("step %s: check %s" % ((yi, xi), check))

    def resume_init(self, init_train=False):
        sess = tf.Session()
        self.sess = sess
        with self.sess.as_default():
            self.init_opt(init=True)
            if init_train:
                self.init_opt(init=False, eval=False, opt_off=True)
            self.init_opt(init=False, eval=True)
            saver = tf.train.Saver()
            if self.resume_from is not None:
                # print("not resuming")
                print("resuming from %s" % self.resume_from)
                # fn = tf.train.latest_checkpoint(self.resume_from)
                # print("latest ckpt: %s" % fn)
                saver.restore(sess, self.resume_from)
                print("resumed")

    def vis(self,init=True):
        if init:
            self.resume_init(init_train=True)
        sess = self.sess
        self.sess.as_default().__enter__()

        feed = self.prepare_feed(self.dataset.validation, self.true_batch_size)

        dist = self.model.output_dist
        bs_per_gpu = self.batch_size // self.num_gpus

        x_var, context_var, proposal_sym = dist.sample_sym(
            bs_per_gpu, unconditional=self.unconditional
        )
        batch_imshp = [bs_per_gpu, ] + list(self.model.image_shape)
        h, w = self.model.image_shape[:2]
        import scipy
        def custom_imsave(name, img):
            assert img.ndim == 3
            if img.shape[2] == 3:
                img = (img + 0.5) * 256
            elif img.shape[2] == 1:
                img = img.reshape(img.shape[:2])
            scipy.misc.imsave(
                name,
                img
            )

        # # samples
        z_var = self.model.latent_dist.sample_prior(bs_per_gpu)
        _, x_dist_info = self.model.decode(z_var, sample=False)
        context = sess.run(x_dist_info)
        if self.unconditional:
            context["context"] = np.zeros_like(context["context"])
        fol = "%s/samples/" % self.checkpoint_dir
        try:
            import os
            os.makedirs(fol)
        except:
            pass
        # originals = tuple(feed.items())[0][1].reshape([128,32,32,3])
        # cur_zeros = np.copy(self.dataset.train.images[:128].reshape([-1,32,32,3]))
        cur_zeros = np.zeros(batch_imshp)
        for yi in range(h):
            for xi in range(w):
                ind = xi + yi * w
                if (ind % 16 == 1):
                    for ix in range(bs_per_gpu):
                        # scipy.misc.imsave(
                        #     "%s/%s_step%s.png" % (fol, ix, ind),
                        #                   (cur_zeros[ix] + 0.5) * 255)
                        custom_imsave(
                            "%s/%s_step%s.png" % (fol, ix, ind),
                            cur_zeros[ix]
                        )
                proposal_zeros = sess.run(proposal_sym, {
                    x_var: cur_zeros,
                    context_var: context['context'],
                })
                cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :].copy()
        os.system(
            "tar -zcvf %s/../samples.tar.gz %s" % (fol, fol)
        )
        from sandbox.pchen.InfoGAN.infogan.misc.imported import plotting
        img_tile = plotting.img_tile(cur_zeros, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title=None, )
        plotting.plt.savefig("%s/sample_summary.png" % self.checkpoint_dir)
        plotting.plt.close('all')

        # img_tile = plotting.img_tile(cur_zeros, aspect_ratio=1.0, border_color=1.0, stretch=True)
        # img = plotting.plot_img(img_tile, title=None)
        # plotting.plt.savefig("%s/summary.png" % fol)
        # plotting.plt.close('all')
        # # import ipdb; ipdb.set_trace()

        # decompress
        originals = tuple(feed.items())[0][1][-bs_per_gpu:].reshape(batch_imshp)
        # context = sess.run(self.sym_vars["train"]["x_dist_info"], feed)
        cond = sess.run(self.sym_vars["train"]["cond_feats"], feed)
        fol = "%s/decomp/" % self.checkpoint_dir
        try:
            import os
            os.makedirs(fol)
        except:
            pass
        # originals = tuple(feed.items())[0][1].reshape([128,32,32,3])
        # cur_zeros = np.copy(self.dataset.train.images[:128].reshape([-1,32,32,3]))
        for ix in range(bs_per_gpu):
            custom_imsave(
                "%s/%s_step%s.png" % (fol, ix, 0),
                originals[ix]
            )
        cur_zeros = np.zeros_like(originals)
        for yi in range(h):
            for xi in range(w):
                ind = xi + yi * w
                if (ind % 8 == 1):
                    for ix in range(bs_per_gpu):
                        custom_imsave(
                            "%s/%s_step%s.png" % (fol, ix, ind),
                            cur_zeros[ix]
                        )
                proposal_zeros = sess.run(proposal_sym, {
                    x_var: cur_zeros,
                    context_var: cond,
                })
                cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :].copy()
        os.system(
            "tar -zcvf %s/../decomp.tar.gz %s" % (fol, fol)
        )
        interleaved = np.concatenate(
            [originals[np.newaxis, :, :, :, :], cur_zeros[np.newaxis, :, :, :, :]]
        ).transpose([1, 0, 2, 3, 4]).reshape([-1, 32, 32, 3])
        img_tile = plotting.img_tile(interleaved, aspect_ratio=1., border_color=1., stretch=True)
        img = plotting.plot_img(img_tile, title=None, )
        plotting.plt.savefig("%s/decomp_summary.png" % self.checkpoint_dir)
        plotting.plt.close('all')

        import IPython; IPython.embed()

