from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import tf_go
from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset
import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn

dataset = Cifar10Dataset()
flat_dim = dataset.image_dim

noise = Gaussian(flat_dim)
shape = [-1, 16, 16, 12]
shaped_noise = ReshapeFlow(
    noise,
    forward_fn=lambda x: tf.reshape(x, shape),
    backward_fn=lambda x: tf.reshape(x, noise.dim),
)

# 4 checkerboard flows
# note: unbalanced receptive growth version
def resnet_blocks_gen(blocks=4, filters=64):
    def go(x):
        chns = int_shape(x)[3]
        x = nn.conv2d(filters)
        for _ in range(blocks):
            x = nn.gated_resnet(x)
        temp = nn.conv2d(chns*2)
        mu = temp[:,:,:,chns:]
        logstd = tf.tanh(temp[:,:,:,:chns]) # might want learn scaling
        return mu, logstd
    return go
def checkerboard_condition_fn_gen(bin=0, h_collapse=True):
    id = bin % 2
    def split_gen(bit):
        def go(x):
            shp = int_shape(x)
            half = (
                tf_go(x).
                    transpose([0, 3, 1, 2]).
                    reshape([shp[0], shp[3], shp[1]*shp[2]//2, 2]).
                    transpose([0, 2, 1, 3]).
                    value[:,:,:,bit]
            )
            if h_collapse:
                return tf.reshape(
                    half,
                    [shp[0], shp[1], shp[2]//2, shp[3]]
                )
            else:
                raise NotImplementedError
    def merge(condition, effect):
        shp = int_shape(condition)
        if h_collapse:
            vs = [
                tf_go(x).
                    transpose([0, 3, 1, 2]).
                    reshape([shp[0], shp[3], shp[1], shp[2], 1]).
                    value
                for x in (
                    [condition, effect]
                    if id == 0 else [effect, condition]
                )
            ]
            return (
                tf_go(tf.concat(4, vs)).
                    reshape([shp[0], shp[3], shp[1], shp[2]*2]).
                    tranpose([0, 2, 3, 1]).
                    value
            )
        else:
            raise NotImplementedError

    return split_gen(id), split_gen((id + 1) % 2), merge
cur = shaped_noise
for i in range(4):
    cf, ef, merge = checkerboard_condition_fn_gen(i, True) # fixme: for now
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
    backward_fn=lambda x: tf.space_to_depth(x, 2),
)

#  then 3 channel-wise shearing (note, early noise is not extracted)
def channel_condition_fn_gen(bin=0):
    id = bin % 2
    def split_gen(bit):
        def go(x):
            shp = int_shape(x)
            cut = shp[3] // 2
            return x[:,:,:,:cut] if bit == 0 else x[:,:,:,cut:]
    def merge(condition, effect):
        vs = [condition, effect]
        if id != 0:
            vs = reversed(vs)
        return tf.concat(3, vs)
    return split_gen(id), split_gen((id + 1) % 2), merge
cur = upsampled
for i in range(4):
    cf, ef, merge = channel_condition_fn_gen(i, )
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

# another 3 checkerboard
for i in range(3):
    cf, ef, merge = checkerboard_condition_fn_gen(i, True) # fixme: for now
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )


