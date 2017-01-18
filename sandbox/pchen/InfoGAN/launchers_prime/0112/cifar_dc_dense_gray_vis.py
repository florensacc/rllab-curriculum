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

# try smaller receptive field, unconditional resnets=3 seems to do very well on its own
# resume above with larger llr^

### try tiny receptive field: left 3 up 4
# but maintain a good amount of computation

# try repetiviely shortcircuted gated pixelcnn to expand parameters rather than extranins
# larger pixelcnn

# deeper pixelcnn

# [0,0,1,1] grows faster in the beginning but approaching the same point as [0]*4.
# meaning vae part becomes the bottleneck

# try deeper iaf & af

# try kl annealing instead

# try a much bigger vae
# deeper

# deep cond

# straight from 0.025 fails to use code; optimization still unstable

# overfitting too much on the encoder end (decoder probably fine.?)

# more free bits

# high reps problem hard to optimize? try dense net


from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.algos.share_vae import ShareVAE
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, Anneal
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
        return [0.004, ] #0.001]

    @variant
    def seed(self):
        return [42, ]

    @variant
    def zdim(self):
        return [1024, ]#[12, 32]
        # return [256*2, ]#[12, 32]

    @variant
    def min_kl(self):
        return [0.1]# 0.1]
    #
    @variant(hide=False)
    def network(self):
        yield "shared_spatial_code_dense_block"
        # yield "pixelcnn_based_shared_spatial_code"
        # yield "pixelcnn_based_shared_spatial_code_tiny"
        # yield "dummy"

    @variant(hide=False)
    def base_filters(self, ):
        return [18]

    @variant(hide=False)
    def dec_init_size(self, ):
        return [4]

    @variant(hide=False)
    def k(self, num_gpus):
        return [1, ]

    @variant(hide=False)
    def num_gpus(self):
        return [1]

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
        yield 3000

    @variant(hide=True)
    def anneal_after(self, max_epoch):
        return [None]

    @variant(hide=False)
    def context_dim(self, base_filters):
        # return [base_filters]
        # return [32]
        return [64]

    @variant(hide=False)
    def cond_rep(self, context_dim):
        return [context_dim]

    @variant(hide=False)
    def ar_nr_resnets(self, num_gpus):
        return [
            (1,)
        ]

    @variant(hide=False)
    def ar_nr_cond_nins(self, num_gpus):
        return [
            1,
        ]

    @variant(hide=False)
    def ar_nr_extra_nins(self, num_gpus):
        return [
            # [0,0], # 1min15s, 660k infer params
            # [0,0,0], # 1min40s, 1M infer params
            [0,0,1,2,3],
            # [0,0,1,1,]
            # [1,]*7
        ]

    @variant
    def enc_tie_weights(self):
        return [True, ]

    @variant
    def unconditional(self):
        return [False, ]

    @variant(hide=False)
    def nar(self, i_nar):
        return [6,]

    @variant(hide=False)
    def nr(self, unconditional):
        return [2,]

    @variant(hide=False)
    def rep(self, unconditional):
        return [3,]

    # @variant(hide=False)
    # def ar_nr_extra_nins(self, num_gpus):
    #     return [
    #         2,
    #     ]
    #
    # @variant
    # def enc_tie_weights(self):
    #     return [True, ]
    #
    # @variant
    # def unconditional(self):
    #     return [False]
    #
    # @variant(hide=False)
    # def nar(self, unconditional):
    #     return [6,]
    #
    # @variant(hide=False)
    # def nr(self, zdim, base_filters):
    #     return [8]
    #
    # @variant(hide=False)
    # def rep(self, unconditional):
    #     return [3]

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
                data_init_scale=0.01,
                linear_context="linear" in v["i_context"],
                gating_context="gating" in v["i_context"],
                share_context=True,
                img_shape=[8,8,zdim//64],
                mean_only=True,
            )

        pixelcnn = CondPixelCNN(
            nr_resnets=v["ar_nr_resnets"],
            nr_filters=v["context_dim"],
            nr_cond_nins=v["ar_nr_cond_nins"],
            nr_extra_nins=v["ar_nr_extra_nins"],
            extra_compute=False,
            grayscale=True,
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
            unconditional=v["unconditional"],
            # kl_coeff=0. if v["unconditional"] else 1,
            # kl_coeff_spec=Anneal(start=0.001, end=1.0, length=60),
            adaptive_kl=True,
            ema_kl_decay=0.95,
            deep_cond=True,
            min_kl_coeff=0.00001,
            resume_from="/home/peter/rllab-private/data/local/0110-cifar-dc-dense-gray-really/0110_cifar_dc_dense_gray_really_2017_01_10_18_54_29_0001/pa_mnist_ar_nr_cond__240000.ckpt",
            # resume_from="data/local/1019-SRF-real-FAR-small-vae-share-lvae-play/1019_SRF_real_FAR_small_vae_share_lvae_play_2016_10_19_20_54_27_0001"
            # staged=True,
            # resume_from="/home/peter/rllab-private/data/local/play-0916-apcc-cifar-nml3/play_0916_apcc_cifar_nml3_2016_09_17_01_47_14_0001",
            # img_on=True,
            # summary_interval=200,
            # resume_from="/home/peter/rllab-private/data/local/play-0917-hybrid-cc-cifar-ml-3l-dc/play_0917_hybrid_cc_cifar_ml_3l_dc_2016_09_18_02_32_09_0001",
        )

        run_experiment_lite(
            algo.vis(),
            exp_prefix="0112_cifar_dc_dense_gray_really_vis",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
        )

