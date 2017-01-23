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

from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import cloudpickle

class VG(VariantGenerator):
    @variant
    def hybrid_collapse(self):
        return [
            True, False
        ]

    @variant
    def seed(self):
        return [42,]

def run_task(v):
    hybrid = v["hybrid_collapse"]

    dataset = Cifar10Dataset(dequantized=True)
    flat_dim = dataset.image_dim

    noise = Gaussian(flat_dim)
    shape = [-1, 8, 8, 12*4]
    shaped_noise = ReshapeFlow(
        noise,
        forward_fn=lambda x: tf.reshape(x, shape),
        backward_fn=lambda x: tf_go(x).reshape([-1, noise.dim]).value,
    )

    # 3 checkerboard flows
    # note: unbalanced receptive growth version
    cur = shaped_noise
    for i in range(3):
        cf, ef, merge = checkerboard_condition_fn_gen(i, (i<2) if hybrid else True)
        cur = ShearingFlow(
            cur,
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    #  then 3 channel-wise shearing (note, early noise is not extracted)
    for i in range(3):
        cf, ef, merge = channel_condition_fn_gen(i, )
        cur = ShearingFlow(
            cur,
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

    # another 3 checkerboard
    for i in range(3):
        cf, ef, merge = checkerboard_condition_fn_gen(i, (i<2) if hybrid else True)
        cur = ShearingFlow(
            cur,
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    for i in range(3):
        cf, ef, merge = channel_condition_fn_gen(i, )
        cur = ShearingFlow(
            cur,
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
            cur,
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    dist = DequantizedFlow(cur)

    algo = DistTrainer(
        dataset=dataset,
        dist=dist,
        init_batch_size=1024,
        train_batch_size=64, # also testing resuming from diff bs
        optimizer=AdamaxOptimizer(learning_rate=1e-3),
        save_every=20,
        # resume_from="/home/peter/rllab-private/data/local/global_proper_deeper_flow/"
        # checkpoint_dir="data/local/test_debug",
    )
    algo.train()

variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="vh_collapse_flow",
        variant=v,
        # mode="local_docker",
        # env=dict(
        #     CUDA_VISIBLE_DEVICES="5"
        # ),
        mode="ec2",
        use_gpu=True,
        snapshot_mode="last",
        docker_image="dementrock/rllab3-shared-gpu-cuda80",
        seed=v["seed"]
    )
    break
