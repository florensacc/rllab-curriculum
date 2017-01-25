# end itr:
# 2017-01-23 09:42:58.908670 PST | -------------------------  ----------                                                                                                                                    │····························································
# 2017-01-23 09:42:58.908854 PST | train_bits_per_dimAverage  -3.29072                                                                                                                                      │····························································
# 2017-01-23 09:42:58.908938 PST | train_bits_per_dimStd       0.0492968                                                                                                                                    │····························································
# 2017-01-23 09:42:58.909008 PST | train_bits_per_dimMedian   -3.29138                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909097 PST | train_bits_per_dimMin      -3.47683                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909194 PST | train_bits_per_dimMax      -3.15444                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909270 PST | vali_bits_per_dimAverage   -3.45691                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909345 PST | vali_bits_per_dimStd        0.0543612                                                                                                                                    │····························································
# 2017-01-23 09:42:58.909410 PST | vali_bits_per_dimMedian    -3.45021                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909506 PST | vali_bits_per_dimMin       -3.58544                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909563 PST | vali_bits_per_dimMax       -3.34813                                                                                                                                      │····························································
# 2017-01-23 09:42:58.909622 PST | -------------------------  ----------

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
    cf, ef, merge = checkerboard_condition_fn_gen(i, i<2) # fixme: for now
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(blocks=8),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

#  then 3 channel-wise shearing (note, early noise is not extracted)
for i in range(3):
    cf, ef, merge = channel_condition_fn_gen(i, )
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(blocks=8),
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
    cf, ef, merge = checkerboard_condition_fn_gen(i, i<2) # fixme: for now
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(blocks=8),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

for i in range(3):
    cf, ef, merge = channel_condition_fn_gen(i, )
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(blocks=8),
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
    cf, ef, merge = checkerboard_condition_fn_gen(i, i < 2) # fixme: for now
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(blocks=8),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

dist = OldDequantizedFlow(cur)

fol = "data/local/nndeeper_vcollapse_global_proper_deeper_flow"
logger.set_snapshot_dir(fol)

algo = DistTrainer(
    dataset=dataset,
    dist=dist,
    init_batch_size=1024,
    train_batch_size=128, # also testing resuming from diff bs
    optimizer=AdamaxOptimizer(learning_rate=2e-3),
    save_every=20,
    # resume_from="/home/peter/rllab-private/data/local/global_proper_deeper_flow/"
    # checkpoint_dir="data/local/test_debug",

)
algo.train()