

from distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli

import tensorflow as tf
import os
from datasets import MnistDataset, FaceDataset
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

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = ""#now.strftime('%Y_%m_%d_%H_%M_%S')

root_log_dir = "logs/mnist_cat"#_{timestamp}".format(timestamp=timestamp)
root_checkpoint_dir = "ckt/mnist_cat"#_{timestamp}".format(timestamp=timestamp)
batch_size = 128
updates_per_epoch = 100
max_epoch = 1000#1000


class VG(VariantGenerator):
    @variant
    def learning_rate(self):
        # yield 1e-3
        # yield 5e-4#1e-3
        return [5e-4, 1e-4]

    @variant
    def info_reg_coeff(self):  # , use_info_reg):
        # yield 0.1
        return [1.0]#, 0.1, 0.01, 0.001]
        # yield 0.1#.1
        # if use_info_reg:
        #     return [1.0, 0.1, 0.01]#, 0.001]
        # else:
        #     return [0.]

    @variant#(hide=True)
    def use_separate_recog(self):
        """
        Whether to use a separate network for predicting the categorical distribution
        when forming the MI term
        """
        # yield True # True
        return [False]#True, False]

    @variant
    def recog_reg_coeff(self):
        # yield 1.0  # 0.1
        return [0.1, 0.01]#1.0, 0.1, 0.01, 0.001]

    @variant(hide=True)
    def seed(self):
        # yield 21  # 1
        return [1, 11, 21, 31, 41]

    @variant
    def network(self):
        yield "large_conv"


vg = VG()

variants = vg.variants()

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

        dataset = FaceDataset()

        # data_directory = "MNIST"
        # if not os.path.exists(data_directory):
        #     os.makedirs(data_directory)
        # mnist = input_data.read_data_sets(data_directory, one_hot=True)

        latent_spec = [
            # (Uniform(128), False),
            (Gaussian(128), False),
            (Gaussian(1), True),
            # (Categorical(10), True),
        ]

        # model = RegularizedHelmholtzMachine(
        #     output_dist=MeanBernoulli(28 * 28),
        #     latent_spec=latent_spec,
        #     batch_size=batch_size,
        #     network_type=v["network"]
        # )

        model = RegularizedGAN(
            output_dist=MeanBernoulli(dataset.image_dim),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            use_separate_recog=True,
            network_type=v["network"]
        )

        algo = RegularizedVAE(

        )

        algo = InfGANTrainer(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
            # info_reg_coeff=0.0,#,v["info_reg_coeff"],
            info_reg_coeff=1.0,#0.0,#,v["info_reg_coeff"],
            generator_learning_rate=1e-4,
            discriminator_learning_rate=1e-4,
        )

        algo.train()
