
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import dateutil
import datetime
import numpy as np
import sys

from rllab.misc.instrument import VariantGenerator, variant
from sandbox.pchen.InfoGAN.infogan.algos.infogan_trainer import InfoGANTrainer
from sandbox.pchen.InfoGAN.infogan.misc.datasets import ChairDataset
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Gaussian, Categorical, Bernoulli, MeanBernoulli
from sandbox.pchen.InfoGAN.infogan.misc.utils import skip_if_exception, mkdir_p, set_seed
from sandbox.pchen.InfoGAN.infogan.models.regularized_gan import RegularizedGAN

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

name = "1129_chair"
root_log_dir = "logs/" + name
root_checkpoint_dir = "ckt/" + name
batch_size = 128
updates_per_epoch = 100
# max_epoch = 20


dataset = ChairDataset()


class VG(VariantGenerator):
    @variant
    def max_epoch(self):
        return [100,]

    @variant
    def d_lr(self):
        return [1e-4]
        # yield 2e-4
        # yield 1e-3
        # yield 2e-4
        # yield 1e-4
        # return np.arange(1, 11) * 1e-4#return [1e-4, 2e-4]
        # yield 6e-4
        # yield 2e-4
        # return np.arange(1, 11) * 1e-4#[1e-3, 5e-4, 1e-4]

    @variant
    def g_lr(self):
        return [1e-3]
        # yield 1e-3#6e-4
        # yield 2e-4
        # yield 1e-3
        # return np.arange(1, 11) * 1e-4
        # yield 1e-3
        # yield 5e-4
        # yield 1e-3
        # return np.arange(1, 11) * 1e-4#[1e-3, 5e-4, 1e-4]
        # return [1e-3, 5e-4, 1e-4]

    @variant
    def info_reg_coeff(self):
        # yield 1.0
        # 1.0
        return [0.01, ]
        # return [1.0, 5.0, 10.0, 50.0, 100.0]
        # return [1.0, 1.5, 2.0, 2.5, 5.0, 10.0]
        # return np.arange(1, 11) * 0.1
        # yield 0.3


    @variant
    def cont_info_reg_coeff(self):
        # yield None
        return [1.0, ]
        # yield 1.0#0.1
        # return np.arange(1, 11) * 0.01
        # return [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        # yield 0.05
        # yield 1.0#0.2
        # yield 1.0
        # return list(np.arange(1, 11) * 0.1)

    @variant
    def use_separate_recog(self):
        """
        Whether to use a separate network for predicting the categorical distribution
        when forming the MI term
        """
        # yield False
        # yield True#False
        return [False]
        # yield False#True
        # yield True#False#True#True
        # yield False
        # return [True, False]

    @variant
    def cont_dist(self):
        yield "uniform"
        # return ["gaussian", "uniform"]
        # yield "gaussian"
        # yield "gaussian"
        # return ["uniform", "gaussian"]

    @variant
    def n_cont(self):
        # yield 2#8
        # yield 2
        yield 1#2#1#[0, 1, 2]
        # yield 2#5#2#5#1
        # if n_disc == 0:
        #     return np.arange(1, 11)
        # else:
        #     return np.arange(11)
            # return [0, 3, 5, 7, 10]
    @variant
    def n_bin(self):
        yield 0#10

    @variant
    def n_disc(self):
        yield 3
        # return np.arange(2, 10)
        # yield 2
        # yield 0#2
        # yield 1
        # yield 1
        # return np.arange(7)
        # yield 1
        # return [0, 1, 2, 3]

    @variant
    def reg_epochs(self):
        yield 0#30
        # return [0, 30]
        # yield 30#10

    @variant#(hide=True)
    def seed(self):
        # yield 21
        # yield 21
        # yield 21#31
        # yield 41
        return [42]

    @variant
    def network(self):
        yield "tmp1"
        # yield "tmp1_resize"

    @variant
    def truncate_std(self):
        yield False
        # return [True, False]
        # yield True#False
        # yield True#False#True


vg = VG()

variants = vg.variants(randomized=True)
print(len(variants))

for v in variants:

    # with skip_if_exception():

        tf.reset_default_graph()
        max_epoch = v["max_epoch"]

        exp_name = "chair_%s" % (timestamp)
        for k in ["max_epoch", "d_lr", "g_lr", "network"]:
            exp_name = "%s_%s_%s" % (k, v[k], exp_name)
        try:

            print("Exp name: %s" % exp_name)

            log_dir = os.path.join(root_log_dir, exp_name)
            checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

            mkdir_p(log_dir)
            mkdir_p("failed/%s" % root_log_dir)
            mkdir_p(checkpoint_dir)

            set_seed(v["seed"])

            if v["cont_dist"] == "uniform":
                cont_dist = lambda x: Uniform(x, truncate_std=v["truncate_std"])
            elif v["cont_dist"] == "gaussian":
                cont_dist = lambda x: Gaussian(x, truncate_std=v["truncate_std"])
            else:
                raise NotImplementedError

            latent_spec = [
                (cont_dist(128), False),
            ] + [(Categorical(20), True)] * v["n_disc"] + [(Bernoulli(1), True)] * v["n_bin"] + [(cont_dist(1), True)] * v["n_cont"]
            #     (Categorical(10), True),
            #     # (Gaussian(1), True),
            #     # (Gaussian(1), True),
            #     # (Gaussian(1), True),
            # ]

            model = RegularizedGAN(
                output_dist=MeanBernoulli(dataset.image_dim),
                latent_spec=latent_spec,
                batch_size=batch_size,
                image_shape=dataset.image_shape,
                use_separate_recog=v["use_separate_recog"],
                network_type=v["network"]
            )

            algo = InfoGANTrainer(
                model=model,
                dataset=dataset,
                # scheduled_datasets=datasets,
                batch_size=batch_size,
                exp_name=exp_name,
                log_dir=log_dir,
                checkpoint_dir=checkpoint_dir,
                max_epoch=max_epoch,
                updates_per_epoch=updates_per_epoch,
                snapshot_interval=1000,
                info_reg_coeff=v["info_reg_coeff"],
                # cont_info_reg_coeff=v["cont_info_reg_coeff"],
                generator_learning_rate=v["g_lr"],
                discriminator_learning_rate=v["d_lr"],
                reg_epochs=v["reg_epochs"],
            )

            algo.train()
        except Exception as e:
            print(e)
            # raise
            print("Moving to failed experiments")
            os.system("mv %s failed/%s" % (log_dir, log_dir))
            # raise

