from __future__ import print_function
from __future__ import absolute_import
from distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli

import tensorflow as tf
import os
from datasets import MnistDataset, ChairDataset, FaceDataset
from tensorflow.examples.tutorials.mnist import input_data
from instrument import VariantGenerator, variant
from regularized_helmholtz_machine import RegularizedHelmholtzMachine
from vae import VAE
from regularized_vae import RegularizedVAE
from regularized_gan import RegularizedGAN
from gan_trainer import GANTrainer
from infgan_trainer import InfGANTrainer
from misc import mkdir_p, set_seed, skip_if_exception
import dateutil
import datetime
import numpy as np
import sys

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = ""#now.strftime('%Y_%m_%d_%H_%M_%S')

root_log_dir = "logs/face_gan_gray_enough_snapshot"
root_checkpoint_dir = "ckt/face_gan_gray_enough_snapshot"
batch_size = 128
updates_per_epoch = 10
max_epoch = 2000#1000


class VG(VariantGenerator):
    @variant
    def d_lr(self):
        # yield 6e-4
        # yield 2e-4
        return np.arange(1, 11) * 1e-4#[1e-3, 5e-4, 1e-4]

    @variant
    def g_lr(self):
        # yield 5e-4
        # yield 1e-3
        return np.arange(1, 11) * 1e-4#[1e-3, 5e-4, 1e-4]
        # return [1e-3, 5e-4, 1e-4]

    @variant
    def info_reg_coeff(self):
        # yield 1.0#0.2
        # yield 1.0
        # yield 1.0
        return list(np.arange(1, 11) * 0.1)

    @variant
    def use_separate_recog(self):
        """
        Whether to use a separate network for predicting the categorical distribution
        when forming the MI term
        """
        # yield True#False#True#True
        # yield False
        yield False
        # return [True, False]

    @variant
    def cont_dist(self):
        yield "uniform"
        # return ["uniform", "gaussian"]

    @variant
    def n_cont(self, n_disc):
        yield 5#3#4#3#4
        # yield 2#8
        # yield 2
        # if n_disc == 0:
        #     return np.arange(1, 7)
        # else:
        #     return np.arange(7)
        #     # return [0, 3, 5, 7, 10]

    @variant
    def n_disc(self):
        # yield 1
        # yield 1
        yield 0#[0, 1]
        # yield 1
        # return [0, 1, 2, 3]

    @variant#(hide=True)
    def seed(self):
        # yield 21#31
        # yield 41
        return [1, 11, 21, 31, 41]

    @variant(hide=True)
    def network(self):
        yield "large_conv"


vg = VG()

variants = vg.variants(randomized=True)

dataset = FaceDataset()

for v in variants:

    with skip_if_exception():

        tf.reset_default_graph()
        exp_name = "face_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        log_dir = os.path.join(root_log_dir, exp_name)
        checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

        mkdir_p(log_dir)
        mkdir_p(checkpoint_dir)

        set_seed(v["seed"])

        if v["cont_dist"] == "uniform":
            cont_dist = Uniform
        elif v["cont_dist"] == "gaussian":
            cont_dist = Gaussian
        else:
            raise NotImplementedError

        latent_spec = [
            (cont_dist(128), False),
        ] + [(Categorical(10), True)] * v["n_disc"] + [(cont_dist(1), True)] * v["n_cont"]
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

        algo = InfGANTrainer(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            snapshot_interval=1000,
            updates_per_epoch=updates_per_epoch,
            info_reg_coeff=v["info_reg_coeff"],
            generator_learning_rate=v["g_lr"],
            discriminator_learning_rate=v["d_lr"],
        )

        algo.train()
