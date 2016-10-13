# try removing iaf and af to see if it reduces overfitting

# iaf creates overfitting more. af alone with ard6 -> 79.26
# interestingly, 8 overfits more than 6!

# try tying arconv param & fewer conv ar channels

from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer
from sandbox.pchen.InfoGAN.infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli, Bernoulli, Mixture, AR, \
    IAR, ConvAR

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, load_caltech, Caltech101Dataset
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
batch_size = 128*3
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
        return [64, ]#[12, 32]

    @variant
    def min_kl(self):
        return [0.01, ] #0.05, 0.1]
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
        yield "resv1_k3_pixel_bias_filters_ratio_conv_ar"

    @variant(hide=False)
    def base_filters(self, ):
        return [16, ]

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
    def k(self):
        return [128, ]

    @variant(hide=False)
    def nar(self):
        return [4,]

    @variant(hide=False)
    def nr(self):
        return [5,]

    @variant(hide=False)
    def i_nar(self, nar):
        if nar == 0:
            return [4, ]
        else:
            return [0]

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
        return [0.99, ]

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
    #         "mnist",
    #         "omni"
    #     ]

    @variant(hide=True)
    def max_epoch(self, ):
        yield 600

    @variant(hide=True)
    def anneal_after(self, max_epoch):
        return [int(max_epoch * 0.7)]

    @variant(hide=False)
    def context_dim(self, ):
        return [4]

    @variant(hide=False)
    def cond_rep(self, context_dim):
        return [context_dim]

    @variant(hide=False)
    def ar_depth(self):
        return [6, ]

    @variant(hide=False)
    def ar_nin(self, ar_depth):
        return [2, ]

    @variant(hide=False)
    def ar_tie(self):
        return [True, False]

    @variant(hide=False)
    def ar_chns(self):
        return [12, 6]



vg = VG()

variants = vg.variants(randomized=False)

print(len(variants))

for v in variants[:]:

    # with skip_if_exception():
        max_epoch = v["max_epoch"]

        zdim = v["zdim"]
        import tensorflow as tf
        tf.reset_default_graph()
        exp_name = "pa_mnist_%s" % (vg.to_name_suffix(v))

        print("Exp name: %s" % exp_name)

        # load_caltech()
        # dataset = Caltech101Dataset()
        # dataset = Caltech101Dataset()
        dataset = BinarizedMnistDataset()

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

        ar_conv_dist = ConvAR(
            tgt_dist=MeanBernoulli(1),
            shape=(28, 28, 1),
            filter_size=3,
            depth=v["ar_depth"],
            nr_channels=12,
            pixel_bias=True,
            # block="plstm",
            context_dim=v["context_dim"],
            tieweight=v["ar_tie"],
            block="gated_resnet",
            extra_nins=v["ar_nin"],
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
            vali_eval_interval=6000//128*5,
            exp_avg=v["exp_avg"],
            anneal_after=v["anneal_after"],
            img_on=False,
            vis_ar=False,
        )

        run_experiment_lite(
            algo.train(),
            exp_prefix="1012_play_reg_smnist_cc_af_gres",
            seed=v["seed"],
            variant=v,
            mode="local",
            # mode="lab_kube",
            # n_parallel=0,
            # use_gpu=True,
            # node_selector={
            #     "aws/type": "p2.xlarge",
            #     "openai/computing": "true",
            # },
            # resources=dict(
            #     requests=dict(
            #         cpu=1.6,
            #     ),
            #     limits=dict(
            #         cpu=1.6,
            #     )
            # )
        )


