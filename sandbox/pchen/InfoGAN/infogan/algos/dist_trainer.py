from collections import defaultdict

from pip._vendor.distlib.locators import locate

from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Bernoulli, Gaussian, Mixture, DiscretizedLogistic, ConvAR
import rllab.misc.logger as logger
from rllab.misc.ext import delete
import sys
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, logsumexp, flatten, assign_to_gpu, \
    average_grads, temp_restore, np_logsumexp, get_available_gpus


class DistTrainer(object):
    def __init__(
            self,
            dataset,
            dist,
            optimizer,
            init_batch_size=512,
            train_batch_size=64,
            exp_avg=0.998,
    ):
        self._dist = dist
        self._optimizer = optimizer
        self._dataset = dataset
        self._train_batch_size = train_batch_size
        self._init_batch_size = init_batch_size
        self._gpus = get_available_gpus()
        self._exp_avg = exp_avg

    def init_opt(self, init=False):
        if init:
            self._dist.init_mode()
            batch_size = self._init_batch_size
        else:
            self._dist.train_mode()
            batch_size = self._train_batch_size
        inp_shape = (batch_size,) + self._dataset.image_shape
        x_inp = tf.placeholder(tf.float32, shape=inp_shape)

        cpu_device = "/cpu:0"
        if init:
            with tf.device(cpu_device):
                logprobs = self._dist.logli_prior(x_inp)
        else:
            devices = self._gpus if len(self._gpus) != 0 else [cpu_device]

            tower_grads_lst = []
            for x, device in zip(
                tf.split(0, len(devices), x_inp),
                devices
            ):
                with tf.device(device):
                    logprobs = self._dist.logli_prior(x)
                    tower_loss = -tf.reduce_mean(logprobs)
                    tower_grads = self._optimizer.compute_gradients(tower_loss)
                    tower_grads_lst.append(tower_grads)

        if self._exp_avg is not None:
            self.ema = tf.train.ExponentialMovingAverage(decay=self._exp_avg)
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
        logger.log("opt_inited w/ init=True")

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
                            logger.log("TF init op finished")
                        self.init_opt(init=False, eval=True)
                        logger.log("opt_inited eval=True")
                        self.init_opt(init=False, eval=False)
                        logger.log("opt_inited eval=False")
                        vs = tf.all_variables()
                        sess.run(tf.initialize_variables([
                                                             v for v in vs if
                                                             "optim" in v.name or "global_step" in v.name or \
                                                             ("cv_coeff" in v.name)
                                                             ]))
                        logger.log("vars initd")
                        if self.staged:
                            sess.run(
                                self.staged_cond_mask.assign(0.)
                            )

                        if self.resume_from is not None:
                            # print("not resuming")
                            print("resuming from %s" % self.resume_from)
                            fn = tf.train.latest_checkpoint(self.resume_from)
                            if fn is None:
                                print("cant find latest checkpoint, treating as checkpoint file")
                                fn = self.resume_from
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

                    self.post_iter_hook(
                        **delete(locals(), "self")
                    )
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
            prev = kw["sess"].run(
                self.kl_coeff
            ) if epoch != 0 else None
            desired = spec(epoch)
            if prev != desired:
                kw["sess"].run(
                    self.kl_coeff.assign(
                        desired
                    )
                )



    def iter_hook(self, **kw):
        pass

    def post_iter_hook(
            self,
            log_vals, log_keys, sess, counter,
            **kw
    ):
        if self.adaptive_kl:
            if counter % int(1. / (1.-self.ema_kl_decay)) == 0:
                ema_kl = dict(zip(log_keys, log_vals))["ema_kl"]
                cur_coeff = desired = 0.001
                if ema_kl < self.min_kl:
                    cur_coeff = sess.run(self.kl_coeff)
                    desired = max(cur_coeff * 0.9, self.min_kl_coeff)
                elif ema_kl > self.min_kl * (1. + self.adaptive_kl_tol):
                    cur_coeff = sess.run(self.kl_coeff)
                    desired = min(cur_coeff * 1.1, 1.)
                if not np.allclose(cur_coeff, desired):
                    sess.run(self.kl_coeff_assginment, {self.kl_coeff_assignee: desired})
                    logger.log("\nema_kl:%s adjusting from %s to %s "%(ema_kl, cur_coeff, desired))
            if counter % self.updates_per_epoch == 0:
                cur_coeff = sess.run(self.kl_coeff)
                logger.log("current coeff %s" % cur_coeff)



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
        restorer = temp_restore(sess, self.ema)
        restorer.__enter__()

        # feed = self.prepare_feed(self.dataset.validation, self.true_batch_size)

        dist = self.model.output_dist
        bs_per_gpu = self.batch_size // self.num_gpus

        import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
        x_var, context_var, proposal_sym, tgt_vec_sym = dist.sample_sym(
            bs_per_gpu, unconditional=self.unconditional, deep_cond=self.deep_cond,
        )
        spatial_logli_sym = nn.discretized_mix_logistic(
            x_var*2,
            tgt_vec_sym
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

        from sandbox.pchen.InfoGAN.infogan.misc.imported import plotting

        # import IPython; IPython.embed()

        # decompress
        for di in range(10):
            feed = self.prepare_feed(self.dataset.validation, self.true_batch_size)
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
            plotting.plt.savefig("%s/decomp_summary_%s.png" % (self.checkpoint_dir, di))
            plotting.plt.close('all')

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
        img_tile = plotting.img_tile(cur_zeros, aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title=None, )
        plotting.plt.savefig("%s/sample_summary.png" % self.checkpoint_dir)
        plotting.plt.close('all')

        # img_tile = plotting.img_tile(cur_zeros, aspect_ratio=1.0, border_color=1.0, stretch=True)
        # img = plotting.plot_img(img_tile, title=None)
        # plotting.plt.savefig("%s/summary.png" % fol)
        # plotting.plt.close('all')
        # # import ipdb; ipdb.set_trace()

        import IPython; IPython.embed()

        # beam-search decompression
        try:
            originals = tuple(feed.items())[0][1][-bs_per_gpu:].reshape(batch_imshp)
            # context = sess.run(self.sym_vars["train"]["x_dist_info"], feed)
            all_cond = sess.run(self.sym_vars["train"]["cond_feats"], feed)
            all_imgs = np.zeros_like(originals)
            for img_i in range(32):
                cur_zeros = np.zeros_like(originals)
                cond = np.repeat(all_cond[img_i:img_i+1], bs_per_gpu, axis=0)
                for yi in range(h):
                    for xi in range(w):
                        ind = xi + yi * w
                        proposal_zeros = sess.run(proposal_sym, {
                            x_var: cur_zeros,
                            context_var: cond,
                        })
                        cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :].copy()
                        granularity = 2
                        if ind % granularity == 0:
                            spatial_logli = sess.run(spatial_logli_sym, {
                                x_var: cur_zeros,
                                context_var: cond,
                            })
                            cur_logli = np.mean(spatial_logli.reshape([-1, 32*32])[:, :ind+1], axis=1)
                            bins = 16
                            idx = np.argsort(cur_logli)[::-1][:bins]
                            print(np.mean(cur_logli) / 3 / np.log(2))
                            cur_zeros = np.repeat(cur_zeros[idx], bs_per_gpu // bins, axis=0)
                img_tile = plotting.img_tile(cur_zeros, aspect_ratio=1., border_color=1., stretch=True)
                img = plotting.plot_img(img_tile, title=None, )
                # plotting.plt.savefig("%s/decomp_beam_%s.png" % (self.checkpoint_dir, img_i))
                plotting.plt.savefig("%s/decomp_16beam_%s_gran_%s.png" % (self.checkpoint_dir, img_i, granularity))
                plotting.plt.close('all')
                all_imgs[img_i] = cur_zeros[0]

            interleaved = np.concatenate(
                [originals[np.newaxis, :, :, :, :], all_imgs[np.newaxis, :, :, :, :]]
            ).transpose([1, 0, 2, 3, 4]).reshape([-1, 32, 32, 3])
            img_tile = plotting.img_tile(interleaved, aspect_ratio=1., border_color=1., stretch=True)
            img = plotting.plot_img(img_tile, title=None, )
            plotting.plt.savefig("%s/decomp_beam_searched_abstract.png" % self.checkpoint_dir)
            plotting.plt.close('all')
        except Exception as e:
            import IPython; IPython.embed()

        import IPython; IPython.embed()

        count = 2
        imgss = []
        for _ in range(count):
            cur_zeros = np.zeros(batch_imshp)
            for yi in range(h):
                for xi in range(w):
                    proposal_zeros = sess.run(proposal_sym, {
                        x_var: cur_zeros,
                        context_var: context['context'],
                    })
                    cur_zeros[:, yi, xi, :] = proposal_zeros[:, yi, xi, :].copy()
                print(yi)
            imgss.append(cur_zeros)
        img_tile = plotting.img_tile(np.concatenate(imgss), aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile, title=None, )
        plotting.plt.savefig("%s/sample_summary_big.png" % self.checkpoint_dir)
        plotting.plt.close('all')

    def eval(self, k=128*80, init=True):
        # logprob evaluation
        if init:
            sess = tf.Session()
            self.sess = sess
        else:
            sess = self.sess
        with self.sess.as_default():
            if init:
                self.init_opt(init=True)
                self.init_opt(init=False, eval=True)
                saver = tf.train.Saver()
                if self.resume_from is not None:
                    # print("not resuming")
                    print("resuming from %s" % self.resume_from)
                    # fn = tf.train.latest_checkpoint(self.resume_from)
                    # print("latest ckpt: %s" % fn)
                    saver.restore(sess, self.resume_from)
                    print("resumed")
            # import IPython; IPython.embed()

            # true_vlb = tf.reduce_mean(
            #     logsumexp(tf.reshape(
            #         log_p_x_given_z - (kls if self.kl_coeff != 0. else 0.),
            #         [-1, self.k])),
            # ) - np.log(self.k)
            logpxz, kls = self.sym_vars['eval']["log_p_x_given_z"], self.sym_vars['eval']["kls"]
            logli = logpxz - kls
            eval_input = self.eval_input_tensor
            imgs = self.dataset.validation.images
            total_bs = imgs.shape[0]
            loglis_buffer = np.zeros([total_bs, k], dtype="float32")
            with temp_restore(sess, self.ema):
                for ki in range(k // self.k):
                    start, end = (ki*self.k), ((ki+1)*self.k)
                    progress = ProgressBar()
                    for bi in progress(range(total_bs)):
                        loglis_buffer[bi, start:end] = sess.run(
                            logli,
                            feed_dict={
                                eval_input: imgs[bi:bi+1, :]
                            }
                        )
                    this_logli = np.mean(
                        np_logsumexp(
                            loglis_buffer[:, :end]
                        ) - np.log(end)
                    )
                    logger.log("k=%s, logli=%s, bits/dim=%s" % (
                        end,
                        this_logli,
                        this_logli / self.model.output_dist.effective_dim / np.log(2)
                    ))
