from __future__ import print_function
from __future__ import absolute_import

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset
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
batch_size = 128
updates_per_epoch = 100
max_epoch = 500

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant

# pa_mnist_lr_0.0001_min_kl_0.05_mix_std_0.8_monte_carlo_kl_True_nm_10_seed_42_zdim_64
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
        return [12, 42, 999]
        # return [123124234]

    @variant
    def monte_carlo_kl(self):
        return [True, ]

    @variant
    def zdim(self):
        return [32]#[12, 32]

    @variant
    def min_kl(self):
        return [0.01, ] #0.05, 0.1]
    #
    @variant
    def nar(self):
        # return [0,]#2,4]
        # return [2,]#2,4]
        # return [0,1,]#4]
        return [4,]

    @variant
    def nr(self, nar):
        if nar == 0:
            return [1]
        else:
            return [10, ]

    # @variant
    # def nm(self):
    #     return [10, ]
    #     return [5, 10, 20]

    # @variant
    # def pr(self):
    #     return [True, False]

    @variant(hide=True)
    def network(self):
        # yield "large_conv"
        # yield "small_conv"
        # yield "deep_mlp"
        # yield "mlp"
        # yield "resv1_k3"
        # yield "conv1_k5"
        # yield "small_res"
        # yield "small_res_small_kern"
        # yield "resv1_k3_pixel_bias"
        yield "resv1_k3_pixel_bias_filters_ratio"

    @variant(hide=True)
    def dec_nn(self, network):
        return [True, ]

    @variant(hide=True)
    def enc_nn(self, network):
        return [True, ]

    @variant()
    def enc_rep(self, network):
        return [2,]

    @variant()
    def dec_rep(self, network):
        return [2,]

    @variant()
    def fs(self, network):
        return [2,3,4,]

    @variant(hide=True)
    def dec_fc_keepprob(self, network):
        return [1.,]

    @variant(hide=True)
    def enc_res_keepprob(self, network):
        if network == "resv1_k3_pixel_bias_filters_ratio":
            return [1.,]
        else:
            return [0]

    @variant(hide=True)
    def dec_res_keepprob(self, network):
        return [1., ]

    @variant(hide=True)
    def wnorm(self):
        return [True, ]

    @variant(hide=True)
    def ar_wnorm(self):
        return [True, ]

    @variant(hide=True)
    def k(self):
        return [128, ]

    @variant(hide=True)
    def i_nar(self):
        return [4, ]

    @variant(hide=True)
    def i_nr(self):
        return [20, ]

    @variant(hide=True)
    def i_init_scale(self):
        return [0.1, ]

    @variant(hide=True)
    def i_context(self):
        # return [True, False]
        return [
            # [],
            ["linear"],
            # ["gating"],
            # ["linear", "gating"]
        ]

    @variant(hide=True)
    def anneal_after(self):
        return [800, ]

    @variant(hide=True)
    def exp_avg(self):
        return [0.999, ]
        # return [0.999, 0.99]
        # return [None]

    @variant(hide=True)
    def dataset(self):
        # yield ResamplingBinarizedMnistDataset(
        #     disable_vali=False,
        # )
        yield ResamplingBinarizedOmniglotDataset()

    @variant(hide=True)
    def ac(self):
        return [0.1, ]


vg = VG()

variants = vg.variants(randomized=False)

print(len(variants))

for v in variants[:]:

    # with skip_if_exception():

        zdim = v["zdim"]
        import tensorflow as tf
        tf.reset_default_graph()
        exp_name = "pa_mnist_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        dataset = v["dataset"]
        # dataset = ResamplingBinarizedMnistDataset(
        #     disable_vali=True,
        # )
        # dataset = MnistDataset()

        dist = Gaussian(zdim)
        for _ in xrange(v["nar"]):
            dist = AR(zdim, dist, neuron_ratio=v["nr"], data_init_wnorm=v["ar_wnorm"])

        latent_spec = [
            (
                dist
                ,
                False
            ),
        ]

        inf_dist = Gaussian(zdim)
        for _ in xrange(v["i_nar"]):
            inf_dist = IAR(
                zdim,
                inf_dist,
                neuron_ratio=v["i_nr"],
                data_init_scale=v["i_init_scale"],
                linear_context="linear" in v["i_context"],
                gating_context="gating" in v["i_context"],
            )

        model = RegularizedHelmholtzMachine(
            output_dist=MeanBernoulli(dataset.image_dim),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            network_type=v["network"],
            inference_dist=inf_dist,
            wnorm=v["wnorm"],
            network_args=dict(
                # base_filters=v["base_filters"],
                # fc_size=v["fc_size"],
                # dec_fc_keep_prob=v["dec_fc_keepprob"],
                dec_res_keep_prob=v["dec_res_keepprob"],
                enc_res_keep_prob=v["enc_res_keepprob"],
                enc_nn=v["enc_nn"],
                dec_nn=v["dec_nn"],
                enc_rep=v["enc_rep"],
                dec_rep=v["dec_rep"],
                ac=v["ac"],
                filter_size=v["fs"],
            ),
        )

        algo = VAE(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            exp_name=exp_name[:20],
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
            optimizer_cls=AdamaxOptimizer,
            optimizer_args=dict(learning_rate=v["lr"]),
            monte_carlo_kl=v["monte_carlo_kl"],
            min_kl=v["min_kl"],
            k=v["k"],
            vali_eval_interval=1500*2,
            exp_avg=v["exp_avg"]
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="0819_nn_omni_fs",
            seed=v["seed"],
            variant=v,
            # mode="local",
            mode="lab_kube",
            n_parallel=0,
            use_gpu=True,
        )


