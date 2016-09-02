from __future__ import print_function
from __future__ import absolute_import

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.algos.cv_vae import CVVAE
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset
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
# updates_per_epoch = 100
max_epoch = 625

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
        return [42, ]
        # return [123124234]

    @variant
    def monte_carlo_kl(self):
        return [True, ]

    @variant
    def zdim(self):
        return [32, ]#[12, 32]

    @variant
    def min_kl(self):
        return [0.0, ] #0.05, 0.1]
    #
    @variant
    def nar(self):
        # return [0,]#2,4]
        # return [2,]#2,4]
        # return [0,1,]#4]
        return [5, 0]

    @variant
    def nr(self, nar):
        if nar == 0:
            return [1]
        else:
            return [ 5, ]

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
        yield "resv1_k3_pixel_bias"

    @variant(hide=True)
    def wnorm(self):
        return [True, ]

    @variant(hide=True)
    def ar_wnorm(self):
        return [True, ]

    @variant(hide=False)
    def k(self):
        return [128, ]

    @variant(hide=False)
    def i_nar(self, nar):
        return [0]

    @variant(hide=False)
    def i_nr(self):
        return [20, ]

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
    def anneal_after(self):
        return [300, ]

    @variant(hide=False)
    def anneal_every(self):
        return [75]

    @variant(hide=False)
    def anneal_factor(self):
        return [0.75, ]

    @variant(hide=False)
    def exp_avg(self):
        return [0.999, ]

    @variant(hide=False)
    def share_context(self):
        return [True, ]

    @variant(hide=False)
    def tiear(self):
        # return [False]
        return [False]

    @variant(hide=False)
    def cv(self):
        # return [False]
        return [True, False]

    @variant(hide=False)
    def alpha_update_interval(self, cv):
        if cv:
            return [5, 50, 250]
        return [0]

    @variant(hide=False)
    def alpha_init(self, cv):
        if cv:
            return [1., 0., ]
        return [0]


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

        dataset = ResamplingBinarizedMnistDataset(disable_vali=True)
        # dataset = MnistDataset()

        dist = Gaussian(zdim)
        for _ in xrange(v["nar"]):
            dist = AR(
                zdim,
                dist,
                neuron_ratio=v["nr"],
                data_init_wnorm=v["ar_wnorm"],
                var_scope="AR_scope" if v["tiear"] else None,
            )

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
                share_context=v["share_context"],
                var_scope="IAR_scope" if v["tiear"] else None,
            )

        model = RegularizedHelmholtzMachine(
            output_dist=MeanBernoulli(dataset.image_dim),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
            network_type=v["network"],
            inference_dist=inf_dist,
            wnorm=v["wnorm"],
        )

        go = dict(
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
            vali_eval_interval=1500*4,
            exp_avg=v["exp_avg"],
            anneal_after=v["anneal_after"],
            anneal_every=v["anneal_every"],
            anneal_factor=v["anneal_factor"],
            img_on=False,
        )
        if v["cv"]:
            algo = CVVAE(
                alpha_update_interval=v["alpha_update_interval"],
                alpha_init=v["alpha_init"],
                **go
            )
        else:
            algo = VAE(
                **go
            )

        run_experiment_lite(
            algo.train(),
            exp_prefix="0830_ar_cv",
            seed=v["seed"],
            variant=v,
            # mode="local",
            mode="lab_kube",
            n_parallel=0,
            use_gpu=True,
        )


