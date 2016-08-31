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

zdim = 32
exp_name = "arvae"

print("Exp name: %s" % exp_name)

dataset = ResamplingBinarizedMnistDataset(disable_vali=True)
# dataset = MnistDataset()

dist = Gaussian(zdim)
for _ in xrange(5):
    dist = AR(
        zdim,
        dist,
        neuron_ratio=5,
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
# for _ in xrange(v["i_nar"]):
#     inf_dist = IAR(
#         zdim,
#         inf_dist,
#         neuron_ratio=v["i_nr"],
#         data_init_scale=v["i_init_scale"],
#         linear_context="linear" in v["i_context"],
#         gating_context="gating" in v["i_context"],
#         share_context=v["share_context"],
#         var_scope="IAR_scope" if v["tiear"] else None,
#     )

model = RegularizedHelmholtzMachine(
    output_dist=MeanBernoulli(dataset.image_dim),
    latent_spec=latent_spec,
    batch_size=batch_size,
    image_shape=dataset.image_shape,
    network_type="resv1_k3_pixel_bias",
    inference_dist=inf_dist,
    wnorm=True,
)

algo = CVVAE(
    model=model,
    dataset=dataset,
    batch_size=batch_size,
    exp_name=exp_name,
    max_epoch=max_epoch,
    optimizer_cls=AdamaxOptimizer,
    optimizer_args=dict(learning_rate=0.002),
    monte_carlo_kl=True,
    min_kl=0.0,
    k=1,
    vali_eval_interval=1500*4,
    anneal_after=300,
    anneal_every=75,
    anneal_factor=0.75,
    img_on=False,
)

run_experiment_lite(
    algo.train(),
    exp_prefix="0830_arvae_cv",
    seed=42,
    mode="local",
    # mode="lab_kube",
    # n_parallel=0,
    # use_gpu=True,
)


