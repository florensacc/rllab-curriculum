from __future__ import print_function
from __future__ import absolute_import
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset
from sandbox.pchen.InfoGAN.infogan.misc.instrument import VariantGenerator, variant
from sandbox.pchen.InfoGAN.infogan.models.regularized_gan import RegularizedGAN
from sandbox.pchen.InfoGAN.infogan.algos.infogan_trainer import InfoGANTrainer
from sandbox.pchen.InfoGAN.infogan.algos.infogan_semi_trainer import InfoGANSemiTrainer
from sandbox.pchen.InfoGAN.infogan.misc.utils import mkdir_p, set_seed, skip_if_exception
import numpy as np
import dateutil
import datetime

now = datetime.datetime.now(dateutil.tz.tzlocal())

root_log_dir = "logs/mnist_semi_large"
root_checkpoint_dir = "ckt/mnist_semi_large"
batch_size = 128
updates_per_epoch = 100
max_epoch = 100


class VG(VariantGenerator):
    @variant
    def d_lr(self):
        yield 3e-4
        # return np.arange(1, 11) * 1e-4

    @variant
    def g_lr(self):
        yield 1e-3
        # return np.arange(1, 11) * 1e-4

    @variant
    def info_reg_coeff(self):
        yield 0.7
        # return np.arange(1, 11) * 0.1

    @variant(hide=True)
    def use_separate_recog(self):
        """
        Whether to use a separate network for predicting the categorical distribution
        when forming the MI term
        """
        yield False

    @variant(hide=True)
    def cont_dist(self):
        yield "uniform"

    @variant  # (hide=True)
    def n_cont(self):
        yield 0
        # return [0, 1, 2]

    @variant  # (hide=True)
    def n_disc(self):
        yield 1
        # return [1]

    @variant
    def truncate_std(self):
        yield True
        # return [True, False]

    @variant
    def seed(self):
        yield 11
        # return [1, 11, 21, 31, 41]

    @variant
    def network(self):
        yield "mnist_catgan"
        # return ["mnist_catgan", "mnist_semi"]

    @variant
    def semi_mode(self):
        yield "reg_latent_dist_info"
        # return ["reg_latent_dist_info", "reg_latent_dist_flat", "reg_latent_dist_both"]

    @variant
    def semi_classifier(self):
        yield "svm"
        # return ["svm", "knn", "mlp"]


vg = VG()

variants = vg.variants(randomized=True)

print(len(variants))

for v in variants:

    with skip_if_exception():

        tf.reset_default_graph()
        exp_name = "mnist_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        log_dir = os.path.join(root_log_dir, exp_name)
        checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

        mkdir_p(log_dir)
        mkdir_p(checkpoint_dir)

        set_seed(v["seed"])

        dataset = MnistDataset()

        if v["cont_dist"] == "uniform":
            cont_dist = lambda x: Uniform(x, truncate_std=v["truncate_std"])
        elif v["cont_dist"] == "gaussian":
            cont_dist = lambda x: Gaussian(x, truncate_std=v["truncate_std"])
        else:
            raise NotImplementedError

        latent_spec = [(cont_dist(128), False)] + \
                      [(Categorical(10), True)] * v["n_disc"] + [(cont_dist(1), True)] * v["n_cont"]

        model = RegularizedGAN(
            output_dist=MeanBernoulli(dataset.image_dim),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            use_separate_recog=v["use_separate_recog"],
            network_type=v["network"]
        )

        algo = InfoGANSemiTrainer(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
            info_reg_coeff=v["info_reg_coeff"],
            generator_learning_rate=v["g_lr"],
            discriminator_learning_rate=v["d_lr"],
            semi_interval=100,
            semi_mode=v["semi_mode"],
        )

        algo.train()
