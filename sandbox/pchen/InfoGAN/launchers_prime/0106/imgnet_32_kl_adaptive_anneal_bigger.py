# super big model on image net

# from imgnet_32.py
# eval seems to take up some cycles -> trim it down a lot
# code seems killed -> much more generous freebits

# kl never exceeds 0.02 under expfreebits
# trying kl annealing instead

# annealing ok; but might need bigger model

# fix anneal

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.algos.share_vae import ShareVAE
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, Anneal
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR, ConvAR, DiscretizedLogistic, DistAR, PixelCNN, CondPixelCNN

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset, ImageNet32Dataset
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
batch_size = 32*4
# updates_per_epoch = 100

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def lr(self):
        return [0.002*2, ] #0.001]

    @variant
    def seed(self):
        return [42, ]

    @variant
    def zdim(self):
        return [1024, ]#[12, 32]
        # return [256*2, ]#[12, 32]

    @variant
    def min_kl(self):
        return [0.04, 0.07]# 0.1]
    #
    @variant(hide=False)
    def network(self):
        yield "pixelcnn_based_shared_spatial_code"
        # yield "pixelcnn_based_shared_spatial_code_tiny"
        # yield "dummy"

    @variant()
    def rep(self):
        return [1]

    @variant(hide=False)
    def base_filters(self, ):
        return [32]

    @variant(hide=False)
    def dec_init_size(self, ):
        return [4]

    @variant(hide=False)
    def k(self, num_gpus):
        return [1]
        # return [batch_size // num_gpus, ]

    @variant(hide=False)
    def num_gpus(self):
        return [4]

    # @variant(hide=False)
    # def nr(self, zdim, base_filters):
    #     return [4]
        # return [base_filters // (zdim // 8 // 8 * 2) , ]

    @variant(hide=False)
    def i_nar(self):
        return [0, ]

    @variant(hide=False)
    def i_nr(self):
        return [2,]

    @variant(hide=False)
    def nar(self, ):
        return [6,]

    @variant(hide=False)
    def nr(self, zdim, base_filters):
        return [4]


    @variant(hide=False)
    def i_context(self):
        # return [True, False]
        return [
            [],
            # ["linear"],
            # ["gating"],
            # ["linear", "gating"]
        ]

    @variant(hide=False)
    def exp_avg(self):
        return [0.999, ]

    @variant(hide=True)
    def max_epoch(self, ):
        yield 30000

    @variant(hide=True)
    def anneal_after(self, max_epoch):
        return [None]

    @variant(hide=False)
    def context_dim(self, base_filters):
        return [base_filters*3]
        return [32]
        return [64]

    @variant(hide=False)
    def cond_rep(self, context_dim):
        return [context_dim]

    @variant(hide=False)
    def ar_nr_resnets(self, num_gpus):
        return [
            (1,),
        ]

    @variant(hide=False)
    def ar_nr_cond_nins(self, num_gpus):
        return [
            2,
        ]

    @variant(hide=False)
    def ar_nr_extra_nins(self, num_gpus):
        return [
            # [0,0], # 1min15s, 660k infer params
            # [0,0,0], # 1min10s, 892k infer params
            # [0,0,1,1,1],
            # [1,]*5,
            # [0,]*7,
            # [1,]*6,
            # [1,]*10,
            [1,1,1,4],
        ]

vg = VG()

variants = vg.variants(randomized=False)

print(len(variants))
i = 1
for v in variants[i:i+1]:

    # with skip_if_exception():
        max_epoch = v["max_epoch"]

        zdim = v["zdim"]
        import tensorflow as tf
        tf.reset_default_graph()
        exp_name = "pa_mnist_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        # dataset = Cifar10Dataset()
        dataset = ImageNet32Dataset()

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
                data_init_wnorm=True,
                data_init_scale=0.01,
                linear_context="linear" in v["i_context"],
                gating_context="gating" in v["i_context"],
                share_context=True,
                var_scope=None,
                img_shape=[8,8,zdim//64],
                mean_only=True,
            )

        pixelcnn = CondPixelCNN(
            nr_resnets=v["ar_nr_resnets"],
            nr_filters=v["context_dim"],
            nr_cond_nins=v["ar_nr_cond_nins"],
            nr_extra_nins=v["ar_nr_extra_nins"],
            extra_compute=False,
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
                filter_size=3,
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
            vali_eval_interval=10000 * 3, # 3 epochs per eval roughly
            exp_avg=v["exp_avg"],
            anneal_after=v["anneal_after"],
            img_on=False,
            num_gpus=v["num_gpus"],
            vis_ar=False,
            slow_kl=True,
            unconditional=False,
            # kl_coeff_spec=Anneal(start=0.001, end=1.0, length=15),
            adaptive_kl=True,
            ema_kl_decay=0.9,
            # updates_per_epoch=50,
            # resume_from="data/local/1019-SRF-real-FAR-small-vae-share-lvae-play/1019_SRF_real_FAR_small_vae_share_lvae_play_2016_10_19_20_54_27_0001"
            # staged=True,
            # resume_from="/home/peter/rllab-private/data/local/play-0916-apcc-cifar-nml3/play_0916_apcc_cifar_nml3_2016_09_17_01_47_14_0001",
            # img_on=True,
            # summary_interval=200,
            # resume_from="/home/peter/rllab-private/data/local/play-0917-hybrid-cc-cifar-ml-3l-dc/play_0917_hybrid_cc_cifar_ml_3l_dc_2016_09_18_02_32_09_0001",
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="0105_imgnet_32_FIXanneal_bigger",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
        )

