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

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR, ConvAR, DiscretizedLogistic, DistAR, PixelCNN

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
batch_size = 48*2
# updates_per_epoch = 100

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

class VG(VariantGenerator):
    @variant
    def lr(self):
        # yield 0.0005#
        # yield
        # return np.arange(1, 11) * 1e-4
        # return [0.0001, 0.0005, 0.001]
        return [0.002, ] #0.001]

    @variant
    def seed(self):
        return [42, ]
        # return [123124234]

    @variant
    def monte_carlo_kl(self):
        return [True, ]

    @variant
    def zdim(self):
        return [256, ]#[12, 32]

    @variant
    def min_kl(self):
        return [0.01, ]# 0.1]
    #
    @variant(hide=False)
    def network(self):
        # yield "large_conv"
        # yield "small_conv"
        # yield "deep_mlp"
        # yield "mlp"
        # yield "resv1_k3"
        # yield "conv1_k5
        # yield "small_res"
        # yield "small_res_small_kern"
        # res_hybrid_long_re_real_anneal.pyyield "resv1_k3_pixel_bias"
        # yield "resv1_k3_pixel_bias"
        # yield "resv1_k3_pixel_bias_widegen"
        # yield "resv1_k3_pixel_bias_widegen_conv_ar"
        # yield "resv1_k3_pixel_bias_filters_ratio"
        # yield "resv1_k3_pixel_bias_filters_ratio_32"
        yield "dummy"

    @variant(hide=False)
    def steps(self, ):
        return [3]
    #
    @variant(hide=False)
    def base_filters(self, ):
        return [33, ]

    @variant(hide=False)
    def dec_init_size(self, ):
        return [4]

    @variant(hide=False)
    def rep(self, ):
        return [1, ]

    @variant(hide=True)
    def wnorm(self):
        return [True, ]

    @variant(hide=True)
    def ar_wnorm(self):
        return [True, ]

    @variant(hide=False)
    def k(self, num_gpus):
        return [1, ]

    @variant(hide=False)
    def num_gpus(self):
        # yield 0.0005#
        # yield
        # return np.arange(1, 11) * 1e-4
        # return [0.0001, 0.0005, 0.001]
        # return [2] #0.001]
        return [1] #0.001]

    @variant(hide=False)
    def nar(self):
        return [0, ]

    @variant(hide=False)
    def nr(self):
        return [4,]

    @variant(hide=False)
    def i_nar(self):
        return [0, ]

    @variant(hide=False)
    def i_nr(self):
        return [5,]

    @variant(hide=False)
    def i_init_scale(self):
        return [0.1, ]

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

    @variant(hide=False)
    def tiear(self):
        return [False]
        # return [True, False]

    @variant(hide=False)
    def dec_context(self):
        return [True, ]

    # @variant(hide=False)
    # def ds(self):
    #     return [
    #         # "mnist",
    #         "omni",
    #     ]

    @variant(hide=True)
    def max_epoch(self, ):
        yield 3000

    @variant(hide=True)
    def anneal_after(self, max_epoch):
        return [None]

    @variant(hide=False)
    def context_dim(self, ):
        return [9]

    @variant(hide=False)
    def cond_rep(self, context_dim):
        return [context_dim]

    # @variant(hide=False)
    # def ar_depth(self):
    #     return [12,6,]
    #
    # @variant(hide=False)
    # def ar_filter_size(self, ar_depth):
    #     return [3, 5]
    #
    # @variant(hide=False)
    # def ar_channels(self, ):
    #     return [48]
    #
    # @variant(hide=False)
    # def ar_block(self, ):
    #     return ["plstm", "gated_resnet"]

    @variant(hide=False)
    def ar_nr_resnets(self, num_gpus):
        # smaller net that uses 1 GPU:
        if num_gpus == 1:
            return [
                # (5,),
                # (2,),
                # (2,1,),
                # (2,1,1,1,),

                (3,),
                # (4,),
                # (6,),
                # (2,2,),
            ]
        elif num_gpus == 2:
            # bigger net that needs 2 GPUs
            return [
                (5,5,),
                (5,3,1),
                (3,3,3),
                (2,2,2,2,2),
            ]
        else:
            raise NotImplemented

    @variant(hide=False)
    def ar_nr_extra_nins(self, num_gpus):
        return [3,]



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


        # if v["ds"] == "omni":
        #     dataset = ResamplingBinarizedOmniglotDataset()
        # else:
        #     dataset = ResamplingBinarizedMnistDataset(disable_vali=True)

        dataset = Cifar10Dataset()

        # init_size = v["dec_init_size"]
        # ch_size = zdim // init_size // init_size
        # tgt_dist = Mixture([
        #     (Gaussian(ch_size), 1./v["nm"])
        #     for _ in range(v["nm"])
        # ])
        # dist = ConvAR(
        #     tgt_dist,
        #     shape=(init_size, init_size, ch_size),
        #     depth=v["ar_depth"],
        #     block=v["ar_block"],
        #     nr_channels=ch_size*3,
        #     pixel_bias=True,
        # )
        dist = Gaussian(zdim)
        for _ in range(v["nar"]):
            dist = AR(
                zdim,
                dist,
                neuron_ratio=v["nr"],
                data_init_wnorm=v["ar_wnorm"],
                var_scope="AR_scope" if v["tiear"] else None,
                img_shape=[8,8,zdim//64],
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
        # nml = 7
        # tgt_dist = Mixture(
        #    [(DiscretizedLogistic(3), 1./nml) for _ in range(nml)]
        # )
        # tgt_ar_dist = DistAR(
        #     3,
        #     tgt_dist,
        #     depth=2,
        #     neuron_ratio=6,
        #     linear_context=True,
        # )
        # ar_conv_dist = ConvAR(
        #     # tgt_dist=Bernoulli(1),
        #     # tgt_dist=tgt_dist,
        #     tgt_dist=tgt_ar_dist,
        #     shape=(32, 32, 3),
        #     filter_size=v["ar_filter_size"],
        #     depth=v["ar_depth"],
        #     nr_channels=v["ar_channels"],
        #     pixel_bias=True,
        #     # context_dim=v["context_dim"],
        #     nin=False,
        #     # block="gated_resnet",
        #     block=v["ar_block"],
        #     extra_nins=2
        #     # block="plstm",
        # )
        pixelcnn = PixelCNN(
            nr_resnets=v["ar_nr_resnets"],
            nr_extra_nins=v["ar_nr_extra_nins"],
        )
        model = RegularizedHelmholtzMachine(
            output_dist=pixelcnn,
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            network_type=v["network"],
            inference_dist=inf_dist,
            wnorm=v["wnorm"],
            network_args=dict(
                cond_rep=v["cond_rep"],
                old_dec=False,
                base_filters=v["base_filters"],
                enc_rep=v["rep"],
                dec_rep=v["rep"],
                filter_size=4,
            ),
        )

        algo = VAE(
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
            monte_carlo_kl=v["monte_carlo_kl"],
            min_kl=v["min_kl"],
            k=v["k"],
            vali_eval_interval=1500*3*4,
            exp_avg=v["exp_avg"],
            anneal_after=v["anneal_after"],
            img_on=False,
            num_gpus=v["num_gpus"],
            vis_ar=False,
            slow_kl=True,
            kl_coeff=0.,
            # resume_from="/home/peter/rllab-private/data/local/play-0916-apcc-cifar-nml3/play_0916_apcc_cifar_nml3_2016_09_17_01_47_14_0001",
            # img_on=True,
            # summary_interval=200,
            # resume_from="/home/peter/rllab-private/data/local/play-0917-hybrid-cc-cifar-ml-3l-dc/play_0917_hybrid_cc_cifar_ml_3l_dc_2016_09_18_02_32_09_0001",
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="1018_real_extra_nins_t_pixelcnn_test",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
        )


