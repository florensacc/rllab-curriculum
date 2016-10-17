# inherit from pool_encoder_arch_on_overfit
# test better setting to more symmetric network & try more rep


# more rep -> code not used! this means designing an architecture to make
# sure information propogates is very important as expected!

# train/test don't really get better

# mgpu version and ardepth12

# sparse adamax to deal with instabitlity when nar=2 (not surprising!_


# just test the pixelcnn part

# test tim's better pixelcnn

# 80 epochs to get under 3.10 and appraoching 3.0 in 200 epochs

# scaling down to observe performance difference

# sharing lvae
# fix kl accounting

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.algos.share_vae import ShareVAE
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR, ConvAR, DiscretizedLogistic, DistAR, PixelCNN, CondPixelCNN

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset
from sandbox.pchen.InfoGAN.infogan.models.regularized_helmholtz_machine import RegularizedHelmholtzMachine
from sandbox.pchen.InfoGAN.infogan.algos.vae import VAE
from sandbox.pchen.InfoGAN.infogan.misc.utils import mkdir_p, set_seed, skip_if_exception
import dateutil
import dateutil.tz
import datetime
import numpy as np

now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = ""#now.strftime('%Y_%m_%d_%H_%M_%S')

root_log_dir = "logs/res_comparison_wn_adamax"
root_checkpoint_dir = "ckt/mnist_vae"
# batch_size = 32 * 3
batch_size = 48*4
# updates_per_epoch = 100

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def lr(self):
        return [0.002, ] #0.001]

    @variant
    def seed(self):
        return [42, ]

    @variant
    def zdim(self):
        return [256, ]#[12, 32]
        # return [256*2, ]#[12, 32]

    @variant
    def min_kl(self):
        return [0.01, ]# 0.1]
    #
    @variant(hide=False)
    def network(self):
        yield "pixelcnn_based_shared_spatial_code"

    @variant(hide=False)
    def rep(self, ):
        return [1, ]

    @variant(hide=False)
    def base_filters(self, ):
        return [32, ]

    @variant(hide=False)
    def dec_init_size(self, ):
        return [4]

    @variant(hide=False)
    def k(self, num_gpus):
        return [batch_size // num_gpus, ]

    @variant(hide=False)
    def num_gpus(self):
        return [4]

    @variant(hide=False)
    def nar(self):
        return [2, ]

    @variant(hide=False)
    def nr(self):
        return [6,]

    @variant(hide=False)
    def i_nar(self):
        return [0, ]

    @variant(hide=False)
    def i_nr(self):
        return [5,]

    @variant(hide=False)
    def i_context(self):
        # return [True, False]
        return [
            # [],
            ["linear"],
            # ["gating"],
            # ["linear", "gating"]
        ]

    @variant(hide=False)
    def exp_avg(self):
        return [0.999, ]

    @variant(hide=True)
    def max_epoch(self, ):
        yield 3000

    @variant(hide=True)
    def anneal_after(self, max_epoch):
        return [None]

    @variant(hide=False)
    def context_dim(self, base_filters):
        return [base_filters]
        return [32]
        return [64]

    @variant(hide=False)
    def cond_rep(self, context_dim):
        return [context_dim]

    @variant(hide=False)
    def ar_nr_resnets(self, num_gpus):
        return [
            (3,)
        ]


vg = VG()

variants = vg.variants(randomized=False)

print(len(variants))
i = 0
for v in variants[i:i+1]:

    # with skip_if_exception():
        max_epoch = v["max_epoch"]

        zdim = v["zdim"]
        import tensorflow as tf
        tf.reset_default_graph()
        exp_name = "pa_mnist_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        dataset = Cifar10Dataset()

        dist = Gaussian(zdim)
        for _ in range(v["nar"]):
            dist = AR(
                zdim,
                dist,
                neuron_ratio=v["nr"],
                data_init_wnorm=True,
            )

        latent_spec = [
            (
                dist
                ,
                False
            ),
        ]

        inf_dist = Gaussian(zdim)
        for _ in range(v["i_nar"]):
            inf_dist = IAR(
                zdim,
                inf_dist,
                neuron_ratio=v["i_nr"],
                data_init_scale=v["i_init_scale"],
                linear_context="linear" in v["i_context"],
                gating_context="gating" in v["i_context"],
                share_context=True,
                var_scope="IAR_scope" if v["tiear"] else None,
            )

        pixelcnn = CondPixelCNN(
            nr_resnets=v["ar_nr_resnets"],
            nr_filters=v["context_dim"],
        )

        model = RegularizedHelmholtzMachine(
            output_dist=pixelcnn,
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            network_type=v["network"],
            inference_dist=inf_dist,
            wnorm=True,
            network_args=dict(
                cond_rep=v["cond_rep"],
                base_filters=v["base_filters"],
                enc_rep=v["rep"],
                dec_rep=v["rep"],
            ),
        )

        algo = ShareVAE(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name,
            max_epoch=max_epoch,
            optimizer_cls=AdamaxOptimizer,
            optimizer_args=dict(
                learning_rate=v["lr"],
                beta2_sparse=True,
            ),
            monte_carlo_kl=True,
            min_kl=v["min_kl"],
            k=v["k"],
            vali_eval_interval=1000*5,
            exp_avg=v["exp_avg"],
            anneal_after=v["anneal_after"],
            img_on=False,
            num_gpus=v["num_gpus"],
            vis_ar=False,
            slow_kl=True,
            # resume_from="/home/peter/rllab-private/data/local/play-0916-apcc-cifar-nml3/play_0916_apcc_cifar_nml3_2016_09_17_01_47_14_0001",
            # img_on=True,
            # summary_interval=200,
            # resume_from="/home/peter/rllab-private/data/local/play-0917-hybrid-cc-cifar-ml-3l-dc/play_0917_hybrid_cc_cifar_ml_3l_dc_2016_09_18_02_32_09_0001",
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="1017_FIXKL_share_lvae_play",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
        )


