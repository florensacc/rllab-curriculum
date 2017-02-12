from sandbox.pchen.InfoGAN.infogan.algos.dist_trainer import DistTrainer
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import tf_go, AdamaxOptimizer, average_grads
from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset
import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
import rllab.misc.logger as logger
from sandbox.pchen.InfoGAN.infogan.models.real_nvp import *

dataset = Cifar10Dataset(dequantized=True)
flat_dim = dataset.image_dim

with scopes.arg_scope(
    [resnet_blocks_gen],
    blocks=4, filters=24,
):
    noise = Gaussian(flat_dim)
    shape = [-1, 16, 16, 12]
    shaped_noise = ReshapeFlow(
        noise,
        forward_fn=lambda x: tf.reshape(x, shape),
        backward_fn=lambda x: tf_go(x).reshape([-1, noise.dim]).value,
    )

    # 4 checkerboard flows
    # note: unbalanced receptive growth version
    cur = shaped_noise
    for i in range(8):
        cf, ef, merge = checkerboard_condition_fn_gen(i, True) # fixme: for now
        cur = ShearingFlow(
            normalize_legacy(cur),
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
            effect_fn=ef,
            combine_fn=merge,
        )

    #  then 3 channel-wise shearing (note, early noise is not extracted)
    for i in range(8):
        cf, ef, merge = channel_condition_fn_gen(i, )
        cur = ShearingFlow(
            normalize_legacy(cur),
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
    for i in range(8):
        cf, ef, merge = checkerboard_condition_fn_gen(i, True) # fixme: for now
        cur = ShearingFlow(
            normalize_legacy(cur),
            nn_builder=resnet_blocks_gen(),
            condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

dist = OldDequantizedFlow(cur)

fol = "data/local/smallnn_norm_way_deeper"
logger.set_snapshot_dir(fol)

algo = DistTrainer(
    dataset=dataset,
    dist=dist,
    optimizer=AdamaxOptimizer(learning_rate=2e-3),
    save_every=20,
    init_batch_size=1024,
    train_batch_size=128, # also testing resuming from diff bs
    # checkpoint_dir="data/local/test_debug",

)
algo.train()