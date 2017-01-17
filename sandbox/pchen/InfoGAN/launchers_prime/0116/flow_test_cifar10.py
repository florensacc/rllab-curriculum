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
cur = shaped_noise
for _ in range(4):
    cur = ShearingFlow(

    )

upsample = dict(
    forward_fn=lambda x: tf_go(x).
        reshape([-1, 16, 16, 3, 4]).
        transpose([0, 3, 1, 2, 4]).
        reshape([-1, 3, 32, 32]).
        transpose([0, 2, 3, 1]).
        value,
    backward_fn=lambda x: tf_go(x).
        transpose([0, 3, 1, 2]).
        reshape([-1, 3, 16, 16, 2, 2]).
        transpose([0, 2, 3, 1, 4, 5]).
        reshape([-1, 16, 16, 12]).
        value,
)


