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
import sandbox.pchen.InfoGAN.infogan.misc.imported.plotting as plotting
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, logsumexp, flatten, assign_to_gpu, \
    average_grads, temp_restore, np_logsumexp, get_available_gpus, restore


class DistTrainer(object):
    def __init__(
            self,
            dataset,
            dist,
            optimizer,
            init_batch_size=512,
            train_batch_size=64,
            exp_avg=0.998,
            max_iter=1000,
            updates_per_iter=None,
            eval_every=1,
            save_every=10,
            checkpoint_dir=None,
            resume_from=None,
            debug=False,
            restart_from_nan=False,
            eval_on=False,
    ):
        self._dist = dist
        self._optimizer = optimizer
        self._dataset = dataset
        self._image_shape = dataset.image_shape
        self._train_batch_size = train_batch_size
        self._init_batch_size = init_batch_size
        self._gpus = get_available_gpus()
        self._exp_avg = exp_avg
        self._sym_vars = {}
        self._max_iter = max_iter
        self._updates_per_iter = (
            updates_per_iter if updates_per_iter is not None
            else dataset.train.images.shape[0] // train_batch_size
        )
        self._eval_every = eval_every
        self._save_every = save_every
        self._checkpoint_dir = checkpoint_dir or logger.get_snapshot_dir()
        self._eval_on = eval_on
        assert self._checkpoint_dir, "checkpoint can't be none"
        self._resume_from = resume_from
        self._debug = debug
        self._restart_from_nan = restart_from_nan

    def construct_init(self):
        self._dist.init_mode()
        batch_size = self._init_batch_size
        inp_shape = (batch_size,) + self._dataset.image_shape

        cpu_device = "/cpu:0"
        with tf.device(cpu_device):
            with tf.name_scope("data_init"):
                x_inp = tf.placeholder(tf.float32, shape=inp_shape)
                logprobs = self._dist.logli_prior(x_inp)

        # save relevant sym vars
        self._sym_vars["init"] = dict(locals())

        return x_inp

    def construct_train(self):
        self._dist.train_mode()
        batch_size = self._train_batch_size
        inp_shape = (batch_size,) + self._dataset.image_shape
        x_inp = tf.placeholder(tf.float32, shape=inp_shape)
        dim = np.prod(self._image_shape)
        log_vars = {}

        cpu_device = "/cpu:0"
        devices = self._gpus if len(self._gpus) != 0 else [cpu_device]

        tower_grads_lst = []
        logprobs = []
        xs = tf.split(0, len(devices), x_inp)
        for i, device in enumerate(
            devices
        ):
            with tf.device(device):
                x = xs[i]
                logprob = tf.reduce_mean(self._dist.logli_prior(x))
                logprobs.append(logprob)
                tower_loss = -logprob
                tower_grads = self._optimizer.compute_gradients(tower_loss)
                # # clipping attempt
                # tower_grads = [
                #     (tf.clip_by_value(grad, -1., 1.), var)
                #     for grad, var in tower_grads
                # ]
                # # gradient numerics check
                if self._debug:
                    tower_grads = [
                    (tf.check_numerics(grad, var.name), var)
                    for grad, var in tower_grads
                ]
                tower_grads_lst.append(tower_grads)

        with tf.variable_scope("optim"):
            trainer = self._optimizer.apply_gradients(
                grads_and_vars=average_grads(tower_grads_lst)
            )
            if self._exp_avg is not None:
                ema = tf.train.ExponentialMovingAverage(decay=self._exp_avg)
                ema_applied = ema.apply(tf.trainable_variables())
                with tf.name_scope(None):
                    trainer = tf.group(*[trainer, ema_applied])
            else:
                ema = None

        avg_logprob = tf.reduce_mean(logprobs)
        avg_logprob_per_dim = avg_logprob / dim
        bits_per_dim = avg_logprob_per_dim / np.log(2)

        for var_name in [
            # "avg_logprob",
            # "avg_logprob_per_dim",
            "bits_per_dim",
        ]:
            log_vars[var_name] = locals()[var_name]

        # save relevant sym vars
        self._sym_vars["train"] = dict(locals())

        return x_inp, log_vars, trainer, ema

    def construct_eval(self):
        cpu_device = "/cpu:0"
        devices = self._gpus if len(self._gpus) != 0 else [cpu_device]

        imgs_lst = []
        unit_bs = self._train_batch_size // len(devices)
        import sandbox.pchen.InfoGAN.infogan.misc.distributions as dists
        with tf.name_scope("eval"):
            for i, device in enumerate(
                devices
            ):
                with tf.device(device):
                    with dists.set_current_seed((i+2)**2):
                        imgs_lst += [self._dist.sample_prior(unit_bs)]
            imgs = tf.concat(0, imgs_lst)

        return imgs

    def interact(self, embed=True):
        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True) if self._debug else None
        )

        assert self._resume_from
        init_inp = self.construct_init()
        logger.log("opt_inited w/ init=True")
        train_inp, train_logs, trainer, ema = self.construct_train()
        train_log_names, train_log_vars = zip(*train_logs.items())
        logger.log("opt_inited w/ init=False")
        sample_imgs = self.construct_eval()
        logger.log("opt_inited w/ eval")

        saver = tf.train.Saver()

        with sess.as_default():
            print("resuming from %s" % self._resume_from)
            fn = tf.train.latest_checkpoint(self._resume_from)
            if fn is None:
                print("cant find latest checkpoint, treating as checkpoint file")
                fn = self._resume_from
            saver.restore(sess, fn)
            logger.log("Restore finished")
            if embed:
                import IPython; IPython.embed()
        return locals()


    def train(self):
        sess = tf.Session()

        init_inp = self.construct_init()
        logger.log("opt_inited w/ init=True")

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


        train_inp, train_logs, trainer, ema = self.construct_train()
        train_log_names, train_log_vars = zip(*train_logs.items())
        logger.log("opt_inited w/ init=False")
        if self._eval_on:
            sample_imgs = self.construct_eval()
            logger.log("opt_inited w/ eval")

        saver = tf.train.Saver()

        with sess.as_default():
            if self._resume_from:
                print("resuming from %s" % self._resume_from)
                fn = tf.train.latest_checkpoint(self._resume_from)
                if fn is None:
                    print("cant find latest checkpoint, treating as checkpoint file")
                    fn = self._resume_from
                saver.restore(sess, fn)
                logger.log("Restore finished")
            else:
                init = tf.initialize_all_variables()
                sess.run(init, {init_inp: self.init_batch()})
                logger.log("Init finished")

            if self._debug:
                from tensorflow.python.framework import dtypes
                from tensorflow.python.framework import ops
                from tensorflow.python.ops import array_ops
                from tensorflow.python.ops import control_flow_ops
                check_op = []
                # This code relies on the ordering of ops in get_operations().
                # The producer of a tensor always comes before that tensor's consumer in
                # this list. This is true because get_operations() returns ops in the order
                # added, and an op can only be added after its inputs are added.
                for op in ops.get_default_graph().get_operations():
                    for output in op.outputs:
                        if output.dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
                            if "save" in op.name or "data_init" in op.name or "optim" in op.name or "ExponentialMovingAverage" in op.name or "eval" in op.name:
                                # print("ignoring " + op.name)
                                continue
                            message = op.name + ":" + str(output.value_index) + "; traceback: " + "|||".join(" ".join(str(item) for item in items) for items in op.traceback)
                            with ops.control_dependencies(check_op):
                                check_op = [array_ops.check_numerics(output, message=message)]
                check_op = control_flow_ops.group(*check_op)
                trainer = tf.group(check_op, trainer)
            train_log_vals = []
            for itr in range(self._max_iter):
                for update_i in ProgressBar()(range(self._updates_per_iter)):
                    try:
                        this_batch = self.train_batch()
                        log_vals = sess.run(
                            (trainer,) + train_log_vars,
                            {train_inp: this_batch}
                        )[1:]
                        train_log_vals.append(log_vals)
                        assert not np.any(np.isnan(log_vals))

                    except BaseException as e:
                        print("exception caught: %s" % e)
                        if ("NaN" in e.message) and self._restart_from_nan:
                            restore(sess, ema)
                        else:
                            import IPython; IPython.embed()

                if itr % self._eval_every == 0:
                    # same eval strategy for now
                    vali_log_names = train_log_names
                    vali_log_vals = []
                    with temp_restore(sess, ema):
                        for vali_batch in ProgressBar()(self.vali_batch_gen(itr)):
                            log_vals = sess.run(
                                train_log_vars,
                                {train_inp: vali_batch}
                            )
                            vali_log_vals.append(log_vals)

                        if self._eval_on:
                            samples = sess.run(sample_imgs)
                            # convert to a form that plt likes
                            samples = np.clip(samples+0.5, 0., 1.)
                            img_tile = plotting.img_tile(samples, aspect_ratio=1., border_color=1., stretch=True)
                            _ = plotting.plot_img(img_tile, title=None, )
                            plotting.plt.savefig("%s/samples_itr_%s.png" % (self._checkpoint_dir, itr))
                            plotting.plt.close('all')

                if (itr+1) % self._save_every == 0:
                    fn = saver.save(sess, "%s/%s.ckpt" % (self._checkpoint_dir, itr))
                    logger.log(("Model saved in file: %s" % fn))

                for prefix, ks, ls in [
                    ["train", train_log_names, train_log_vals],
                    ["vali", vali_log_names, vali_log_vals],
                ]:
                    for k, v in zip(ks, np.array(ls).T):
                        # logger.record_tabular("%s_%s" % (prefix, k), np.mean(v))
                        logger.record_tabular_misc_stat("%s_%s" % (prefix, k), (v))
                train_log_vals = []
                logger.log("Iteration #%s" % itr)
                logger.dump_tabular(with_prefix=False)

    def init_batch(self):
        train = self._dataset.train
        x = train.next_batch(self._init_batch_size)[0]
        train.rewind()
        return x.reshape((-1,) + self._image_shape)

    def train_batch(self):
        train = self._dataset.train
        x = train.next_batch(self._train_batch_size)[0]
        return x.reshape((-1,) + self._image_shape)

    def vali_batch_gen(self, itr):
        data = self._dataset.validation
        for _ in range(
            data.images.shape[0] // self._train_batch_size
            if itr != 0 else 3 # extremely rough estimate for first iteration
        ):
            yield data.next_batch(self._train_batch_size)[0].reshape((-1,)+self._image_shape)
        data.rewind()

