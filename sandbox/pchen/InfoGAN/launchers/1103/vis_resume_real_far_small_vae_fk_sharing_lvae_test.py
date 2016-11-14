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
# fix kl accounting & nr_cond_nins sweep

# observe that nr_cond_nins didnt make that much a difference; but 64 featmaps much better than 32

# this experiment explores
# 1. staged training so that the unconditional pixelcnn approximately finishes training before conditional part starts
# 2. varying the number of extra_nins to see if a more powerful pixelcnn is needed

# gain going from 64 -> 92 (0.01)
# some gain from 0 extranin -> 1 extra nin (0.01)
# staging introduced however instability when cond is turned on

# ^ this hence explores no kl

# this is based on the observation that as kl is used more, overfitting is started to be observed.
# try smaller vae

# error: no nar used!

# using nar & experiment with radically smaller min kl

# 0.001 kl shows markedly better performance than 0,01 kl
# 0.001 kl @ 3.0488 and still growing

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
batch_size = 32
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
        return [0.01,0.001]# 0.1]
    #
    @variant(hide=False)
    def network(self):
        # yield "pixelcnn_based_shared_spatial_code"
        yield "pixelcnn_based_shared_spatial_code_tiny"
        # yield "dummy"

    @variant(hide=False)
    def rep(self, ):
        return [1, ]

    @variant(hide=False)
    def base_filters(self, ):
        return [64]

    @variant(hide=False)
    def dec_init_size(self, ):
        return [4]

    @variant(hide=False)
    def k(self, num_gpus):
        return [batch_size // num_gpus, ]

    @variant(hide=False)
    def num_gpus(self):
        return [1]

    @variant(hide=False)
    def nar(self):
        return [4,]

    @variant(hide=False)
    def nr(self, zdim, base_filters):
        return [4]
        # return [base_filters // (zdim // 8 // 8 * 2) , ]

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

    @variant(hide=False)
    def ar_nr_cond_nins(self, num_gpus):
        return [
            1,
        ]

    @variant(hide=False)
    def ar_nr_extra_nins(self, num_gpus):
        return [
            1
        ]

    @variant
    def enc_tie_weights(self):
        return [True, ]


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
                img_shape=[8,8,zdim//64],
                mean_only=True,
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
            nr_cond_nins=v["ar_nr_cond_nins"],
            nr_extra_nins=v["ar_nr_extra_nins"],
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
                enc_tie_weights=v["enc_tie_weights"],
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
            resume_from="data/local/1019-real-FAR-small-vae-share-lvae-play/1019_real_FAR_small_vae_share_lvae_play_2016_10_19_10_04_42_0001/pa_mnist_ar_nr_cond__690000.ckpt"
            # staged=True,
            # resume_from="/home/peter/rllab-private/data/local/play-0916-apcc-cifar-nml3/play_0916_apcc_cifar_nml3_2016_09_17_01_47_14_0001",
            # img_on=True,
            # summary_interval=200,
            # resume_from="/home/peter/rllab-private/data/local/play-0917-hybrid-cc-cifar-ml-3l-dc/play_0917_hybrid_cc_cifar_ml_3l_dc_2016_09_18_02_32_09_0001",
        )

        run_experiment_lite(
            algo.vis(),
            exp_prefix="1103_vis_resume_1019_real_FAR_small_vae_share_lvae_play",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
        )

