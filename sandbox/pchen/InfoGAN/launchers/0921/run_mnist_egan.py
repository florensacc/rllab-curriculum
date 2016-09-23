from sandbox.pchen.InfoGAN.infogan.algos.ensemble_gan_trainer import EnsembleGANTrainer
from sandbox.pchen.InfoGAN.infogan.algos.gan_trainer import GANTrainer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset
from sandbox.pchen.InfoGAN.infogan.models.ensemble_gan import EnsembleGAN
from sandbox.pchen.InfoGAN.infogan.models.gan import GAN
from sandbox.pchen.InfoGAN.infogan.models.regularized_gan import RegularizedGAN
from sandbox.pchen.InfoGAN.infogan.algos.infogan_trainer import InfoGANTrainer
from sandbox.pchen.InfoGAN.infogan.misc.utils import mkdir_p
import dateutil
import datetime
import dateutil.tz

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/mnist"
    root_checkpoint_dir = "ckt/mnist"
    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 100


    nr_model = 2
    exp_name = "real_dps_dls_single_%sensemble_mnist_%s" % (nr_model, timestamp)
    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    dataset = MnistDataset()

    latent_spec = [
        (Uniform(64), False),
        # (Categorical(10), True),
        # (Uniform(1, ), True),
        # (Uniform(1, ), True),
    ]

    glr = 5e-3
    model = EnsembleGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
        nr_models=nr_model,
    )

    algo = EnsembleGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=("glr_%s"%glr)+exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        # info_reg_coeff=1.0,
        generator_learning_rate=glr,
        discriminator_learning_rate=2e-4,
        discriminator_leakage="single",
        discriminator_priviledge="single",
    )

    algo.train()

