from sandbox.pchen.InfoGAN.infogan.algos.dist_trainer import DistTrainer
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import tf_go, AdamaxOptimizer, average_grads
from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset
import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
import rllab.misc.logger as logger
from sandbox.pchen.InfoGAN.infogan.models.real_nvp import *
import random

from rllab import config
from rllab.misc.instrument import run_experiment_lite, VariantGenerator, variant

from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import cloudpickle


# piecewise linear

class VG(VariantGenerator):
    @variant
    def logit(self):
        return [
            True,
        ]

    @variant
    def seed(self):
        return [42,]

def run_task(v):
    logit = v["logit"]
    f = normalize
    hybrid = False

    dataset = Cifar10Dataset(dequantized=False) # dequantization left to flow
    flat_dim = dataset.image_dim

    noise = Gaussian(flat_dim)
    shape = [-1, 16, 16, 12]
    shaped_noise = ReshapeFlow(
        noise,
        forward_fn=lambda x: tf.reshape(x, shape),
        backward_fn=lambda x: tf_go(x).reshape([-1, noise.dim]).value,
    )

    cur = shaped_noise
    for i in range(4):
        cf, ef, merge = checkerboard_condition_fn_gen(i, (i<2) )
        cur = ShearingFlow(
            f(cur),
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    for i in range(4):
        cf, ef, merge = channel_condition_fn_gen(i, )
        cur = ShearingFlow(
            f(cur),
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    # up-sample
    upsampled = ReshapeFlow(
        cur,
        forward_fn=lambda x: tf.depth_to_space(x, 2),
        backward_fn=lambda x: tf_go(x, debug=False).space_to_depth(2).value,
    )
    cur = upsampled
    for i in range(3):
        cf, ef, merge = checkerboard_condition_fn_gen(i, (i<2) if hybrid else True)
        cur = ShearingFlow(
            f(cur),
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    if logit:
        cur = shift(logitize(cur))

    blocks = 4
    filters = 32
    nr_mix = 1
    def go(x):
        shp = int_shape(x)
        chns = shp[3]
        x = nn.conv2d(x, filters)
        for _ in range(blocks):
            x = nn.gated_resnet(x)
        temp = nn.conv2d(x, chns * 2 * nr_mix)
        return tf.reshape(
            temp,
            shp[:3] + [chns*2, nr_mix]
        ) * 0.2 # so slope won't approach 1 so easily
    dist = DequantizedFlow(
        base_dist=cur,
        noise_dist=FactorizedEncodingSpatialPiecewiseLinearDequant(
            shape=[32, 32, 3],
            nn_builder=go,
            nr_mixtures=nr_mix,
        ),
    )

    algo = DistTrainer(
        dataset=dataset,
        dist=dist,
        init_batch_size=1024,
        train_batch_size=256 // 1, # also testing resuming from diff bs
        optimizer=AdamaxOptimizer(
            learning_rate=2e-4,
        ),
        save_every=20,
        # # for debug
        debug=True,
        # resume_from="/home/peter/rllab-private/data/local/global_proper_deeper_flow/"
        # checkpoint_dir="data/local/test_debug",
    )
    algo.train()

variants = VG().variants()

print("#Experiments:", len(variants))

config.AWS_INSTANCE_TYPE = "p2.xlarge"
config.AWS_SPOT = True
config.AWS_SPOT_PRICE = '1.23'
config.AWS_REGION_NAME = 'us-west-2'
config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

for v in variants[:]:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="0202_nf_mixture_fac_encoding_spatial_Plinear_dequnt",
        variant=v,

        mode="local",

        # mode="local_docker",
        # env=dict(
        #     CUDA_VISIBLE_DEVICES="5"
        # ),

        # mode="ec2",
        #
        use_gpu=True,
        snapshot_mode="last",
        docker_image="dementrock/rllab3-shared-gpu-cuda80",
        seed=v["seed"],
        terminate_machine=True,
        # pre_commands=[
        #     "nvidia-modprobe -u -c=0",
        # ],
    )
