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


# to contrast

class VG(VariantGenerator):
    @variant
    def depth(self):
        return [
            0
        ]

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
    for i in range(6):
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
    for i in range(8):
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

    depth = v["depth"]
    if depth == 0:
        dequant_noise = UniformDequant()
    else:
        def shallow_processor(context):
            this = checkerboard_condition_fn_gen()[0](context)
            that = checkerboard_condition_fn_gen()[1](context)
            processed_context = nn.conv2d(tf.concat(3, [this, that]), 32)
            for _ in range(depth):
                processed_context = nn.gated_resnet(processed_context)
            return processed_context
        def flow_builder():
            from sandbox.pchen.InfoGAN.infogan.models.real_nvp import checkerboard_condition_fn_gen
            from sandbox.pchen.InfoGAN.infogan.models.real_nvp import resnet_blocks_gen
            base = Gaussian(flat_dim)
            shape = [32, 32, 3]
            shaped_noise = ReshapeFlow(
                base,
                forward_fn=lambda x: tf.reshape(x, [-1] + list(shape)),
                backward_fn=lambda x: tf_go(x).reshape([-1, base.dim]).value,
            )
            f = normalize
            cur = shaped_noise
            for i in range(depth):
                cf, ef, merge = checkerboard_condition_fn_gen(i, True)
                cur = ShearingFlow(
                    f(cur),
                    nn_builder=resnet_blocks_gen(blocks=2, filters=24),
                    condition_fn=cf,
                    effect_fn=ef,
                    combine_fn=merge,
                )
            return logitize(cur, coeff=256.)
        dequant_noise = FlowBasedDequant(
            shape=[32,32,3],
            context_processor=shallow_processor,
            flow_builder=flow_builder,
        )

    dist = DequantizedFlow(
        base_dist=cur,
        # noise_dist=UniformDequant(),
        noise_dist=dequant_noise,
    )

    algo = DistTrainer(
        dataset=dataset,
        dist=dist,
        init_batch_size=1024,
        train_batch_size=64, # also testing resuming from diff bs
        optimizer=AdamaxOptimizer(
            learning_rate=1e-3,
        ),
        save_every=20,
        # # for debug
        debug=False,
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
        exp_prefix="0211_deep_unif_dequant",
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
