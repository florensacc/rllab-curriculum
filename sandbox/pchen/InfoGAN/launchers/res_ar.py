

from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR

import tensorflow as tf
import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset
from tensorflow.examples.tutorials.mnist import input_data
from sandbox.pchen.InfoGAN.infogan.misc.instrument import VariantGenerator, variant
from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
from sandbox.pchen.InfoGAN.infogan.algos.vae import VAE
from sandbox.pchen.InfoGAN.infogan.algos.regularized_vae import RegularizedVAE
# from regularized_gan import RegularizedGAN
# from gan_trainer import GANTrainer
# from infgan_trainer import InfGANTrainer
from sandbox.pchen.InfoGAN.infogan.misc.utils import mkdir_p, set_seed, skip_if_exception
import dateutil
import dateutil.tz
import datetime
import numpy as np

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = ""#now.strftime('%Y_%m_%d_%H_%M_%S')

root_log_dir = "logs/res_comparison_wn_adamax"
root_checkpoint_dir = "ckt/mnist_vae"
batch_size = 128
updates_per_epoch = 100
max_epoch = 250

# pa_mnist_lr_0.0001_min_kl_0.05_mix_std_0.8_monte_carlo_kl_True_nm_10_seed_42_zdim_64
class VG(VariantGenerator):
    @variant
    def lr(self):
        # yield 0.0005#
        # yield
        # return np.arange(1, 11) * 1e-4
        # return [0.0001, 0.0005, 0.001]
        return [0.0001, 0.001,]

    @variant
    def seed(self):
        return [3389, 98, ]
        # return [123124234]

    @variant
    def monte_carlo_kl(self):
        return [True, ]

    @variant
    def zdim(self):
        return [32, 64]#[12, 32]

    @variant
    def min_kl(self):
        return [0.01, 0.03, 0.1] #0.05, 0.1]
    #
    @variant
    def nar(self):
        # return [0,]#2,4]
        # return [2,]#2,4]
        return [0,1,2,]#4]

    @variant
    def nr(self):
        return [10]

    # @variant
    # def nm(self):
    #     return [10, ]
    #     return [5, 10, 20]

    # @variant
    # def pr(self):
    #     return [True, False]

    @variant(hide=False)
    def network(self):
        # yield "large_conv"
        # yield "small_conv"
        # yield "deep_mlp"
        # yield "mlp"
        yield "small_res"

    @variant(hide=False)
    def wnorm(self):
        return [True, False]


vg = VG()

variants = vg.variants(randomized=True)

print(len(variants))

for v in variants:

    with skip_if_exception():

        zdim = v["zdim"]
        tf.reset_default_graph()
        exp_name = "pa_mnist_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        log_dir = os.path.join(root_log_dir, exp_name)
        checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

        mkdir_p(log_dir)
        mkdir_p(checkpoint_dir)

        set_seed(v["seed"])

        # dataset = BinarizedMnistDataset()
        dataset = MnistDataset()

        dist = Gaussian(zdim)
        for _ in range(v["nar"]):
            dist = AR(zdim, dist, neuron_ratio=v["nr"])

        latent_spec = [
            # (Gaussian(128), False),
            # (Categorical(10), True),
            (
                # Mixture(
                #     [
                #         (
                #             Gaussian(
                #                 zdim,
                #                 # prior_mean=np.concatenate([[2.*((i>>j)%2) for j in xrange(4)], np.random.normal(scale=v["mix_std"], size=zdim-4)]),
                #                 prior_mean=np.concatenate([np.random.normal(scale=v["mix_std"], size=zdim)]),
                #                 init_prior_mean=np.zeros(zdim),
                #                 prior_trainable=True,
                #             ),
                #             1. / nm
                #         ) for i in xrange(nm)
                #     ]
                # )
                dist
                ,
                False
            ),
        ]

        model = RegularizedHelmholtzMachine(
            output_dist=MeanBernoulli(dataset.image_dim),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            network_type=v["network"],
            inference_dist=Gaussian(
                zdim,
            ),
            wnorm=v["wnorm"],
        )

        algo = VAE(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
            # use_separate_recog=v["use_separate_recog"],
            # info_reg_coeff=v["info_reg_coeff"],
            # recog_reg_coeff=v["recog_reg_coeff"],
            learning_rate=v["lr"],
            monte_carlo_kl=v["monte_carlo_kl"],
            min_kl=v["min_kl"],
            # use_prior_reg=v["pr"],
        )

        algo.train()
        # algo.ipdb()

