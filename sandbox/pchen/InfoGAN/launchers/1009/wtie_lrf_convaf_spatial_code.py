# compare results with python rllab/viskit/frontend.py --port 18888 data/local/0927-pool-encoder-arch-on-overfit/

# try playing with conv af model

# kl 0.01 leads to no code being used/
# switch to 0.06 and try again

# 0.06 might need to overfitting, so this will explore 0.01 kl but much smaller init scale
# also try fewer feature maps but this shouldnt affect as much?

# data/local/1003-init-convaf-on-spatial-code/
# --> turns out fewer feature maps helps a lot??
# and only 0.005kl is used

# this explores larger receptive field & convaf

# ar-depth 12 has good performance, this explores training it faster
# w/ multi-gpu and check ar-depth 6 w/ double feat maps & deeper depth

# try to get some best bit by doing param-tying of ar & larger code & deeper and wider AF

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR, ConvAR, DiscretizedLogistic, DistAR

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
# batch_size = 128
batch_size = 129
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
        return [512, ]#[12, 32]

    @variant
    def min_kl(self):
        # return [0.06, ]# 0.1]
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
        # yield "resv1_k3_pixel_bias_filters_ratio_32_global_pool"
        yield "resv1_k3_pixel_bias_filters_ratio_32_big_spatial"

    @variant(hide=False)
    def steps(self, ):
        return [3]
    #
    @variant(hide=False)
    def base_filters(self, ):
        return [32, ]

    @variant(hide=False)
    def dec_init_size(self, ):
        return [4]

    @variant(hide=True)
    def wnorm(self):
        return [True, ]

    @variant(hide=True)
    def ar_wnorm(self):
        return [True, ]

    @variant(hide=False)
    def num_gpus(self):
        # yield 0.0005#
        # yield
        # return np.arange(1, 11) * 1e-4
        # return [0.0001, 0.0005, 0.001]
        return [3] #0.001]

    @variant(hide=False)
    def k(self, num_gpus):
        return [batch_size // num_gpus, ]

    @variant(hide=False)
    def nar(self):
        return [4, ]

    @variant(hide=False)
    def nr(self):
        return [8,]

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
    def i_context(self, i_nar):
        # return [True, False]
        if i_nar == 0:
            return [
                []
            ]
        return [
            [],
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
        return [
            # int(max_epoch * 0.7)
            1500
        ]

    @variant(hide=False)
    def context_dim(self, ):
        return [9]

    @variant(hide=False)
    def cond_rep(self, context_dim):
        return [context_dim]

    @variant(hide=False)
    def ar_depth(self):
        return [12, ]

    @variant(hide=False)
    def data_init_scale(self):
        return [0.01, ]




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
                var_scope="AR_scope",
                img_shape=[8,8,zdim//64],
                data_init_scale=v["data_init_scale"],

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
        nml = 5
        tgt_dist = Mixture(
           [(DiscretizedLogistic(3), 1./nml) for _ in range(nml)]
        )
        tgt_ar_dist = DistAR(
            3,
            tgt_dist,
            depth=2,
            neuron_ratio=2,
            linear_context=True,
        )
        ar_conv_dist = ConvAR(
            # tgt_dist=Bernoulli(1),
            # tgt_dist=tgt_dist,
            tgt_dist=tgt_ar_dist,
            shape=(32, 32, 3),
            filter_size=3,
            depth=v["ar_depth"],
            nr_channels=12*2*2 // 12 * v["ar_depth"],
            pixel_bias=True,
            context_dim=v["context_dim"],
            nin=False,
            block="gated_resnet",
            extra_nins=2
            # block="plstm",
        )
        model = RegularizedHelmholtzMachine(
            output_dist=ar_conv_dist,
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            network_type=v["network"],
            inference_dist=inf_dist,
            wnorm=v["wnorm"],
            network_args=dict(
                cond_rep=v["cond_rep"],
                old_dec=True,
                base_filters=v["base_filters"]
            ),
        )

        algo = VAE(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name,
            max_epoch=max_epoch,
            optimizer_cls=AdamaxOptimizer,
            optimizer_args=dict(learning_rate=v["lr"]),
            monte_carlo_kl=v["monte_carlo_kl"],
            min_kl=v["min_kl"],
            k=v["k"],
            vali_eval_interval=1500*3*4,
            exp_avg=v["exp_avg"],
            anneal_after=v["anneal_after"],
            img_on=False,
            vis_ar=False,
            num_gpus=v["num_gpus"],
            # resume_from="/home/peter/rllab-private/data/local/play-0916-apcc-cifar-nml3/play_0916_apcc_cifar_nml3_2016_09_17_01_47_14_0001",
            # img_on=True,
            # summary_interval=200,
            # resume_from="/home/peter/rllab-private/data/local/play-0917-hybrid-cc-cifar-ml-3l-dc/play_0917_hybrid_cc_cifar_ml_3l_dc_2016_09_18_02_32_09_0001",
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="1009_wtie_mgpu_lrf_convaf_spatial_code",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
        )


