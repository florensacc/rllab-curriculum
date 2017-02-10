from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import tf_go, AdamaxOptimizer, average_grads
from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset
import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
import rllab.misc.logger as logger


@scopes.add_arg_scope_only("blocks", "filters", "squash")
def resnet_blocks_gen(blocks=4, filters=64, squash=tf.tanh):
    def go(x):
        chns = int_shape(x)[3]
        x = nn.conv2d(x, filters)
        for _ in range(blocks):
            x = nn.gated_resnet(x)
        temp = nn.conv2d(x, chns * 2)
        mu = temp[:, :, :, chns:]
        logstd = (temp[:, :, :, :chns])  # might want learn scaling
        if squash:
            logstd = squash(logstd)
        return mu, logstd

    return go


def checkerboard_condition_fn_gen(bin=0, h_collapse=True):
    id = bin % 2

    def split_gen(bit):
        def go(x):
            shp = int_shape(x)
            assert len(shp) == 4
            half = (
                tf_go(x).
                    transpose([0, 3, 1, 2]).
                    reshape([shp[0], shp[3], shp[1] * shp[2] // 2, 2]).
                    transpose([0, 2, 1, 3]).
                    value[:, :, :, bit]
            )
            if h_collapse:
                return tf.reshape(
                    half,
                    [shp[0], shp[1], shp[2] // 2, shp[3]]
                )
            else:
                # collapse vertically
                return (
                    tf_go(half).
                        reshape([shp[0], shp[1]//2, 2, shp[2]//2, shp[3]]).
                        transpose([0, 1, 3, 2, 4]).
                        reshape([shp[0], shp[1]//2, shp[2], shp[3]]).
                        value
                )

        return go

    def merge(condition, effect):
        shp = int_shape(condition)
        assert len(shp) == 4
        xs = [condition, effect] if id == 0 else [effect, condition]
        if not h_collapse:
            xs = [
                tf_go(x).
                    reshape([shp[0], shp[1], shp[2]//2, 2, shp[3]]).
                    transpose([0, 1, 3, 2, 4]).
                    reshape([shp[0], shp[1]*2, shp[2]//2, shp[3]]).
                    value
                for x in xs
            ]
            shp = int_shape(xs[0])
        vs = [
            tf_go(x).
                transpose([0, 3, 1, 2]).
                reshape([shp[0], shp[3], shp[1], shp[2], 1]).
                value
            for x in xs
        ]
        return (
            tf_go(tf.concat(4, vs)).
                reshape([shp[0], shp[3], shp[1], shp[2] * 2]).
                transpose([0, 2, 3, 1]).
                value
        )

    return split_gen(id), split_gen((id + 1) % 2), merge


def channel_condition_fn_gen(bin=0):
    id = bin % 2

    def split_gen(bit):
        def go(x):
            shp = int_shape(x)
            assert len(shp) == 4
            assert shp[3] % 2 == 0
            cut = shp[3] // 2
            return x[:, :, :, :cut] if bit == 0 else x[:, :, :, cut:]

        return go

    def merge(condition, effect):
        vs = [condition, effect]
        if id != 0:
            vs = [effect, condition]
        return tf.concat(3, vs)

    return split_gen(id), split_gen((id + 1) % 2), merge


def main():
    original = (np.random.normal(size=[3,32,32,6]).astype(np.float32))
    test_img = tf.constant(original)
    sess = tf.Session()
    with sess.as_default():
        for i in [0,3,4]:
            cf, ef, merge = checkerboard_condition_fn_gen(i, True)
            target = sess.run(merge(cf(test_img), ef(test_img)))
            assert np.allclose(original, target)
            cf, ef, merge = checkerboard_condition_fn_gen(i, False)
            target = sess.run(merge(cf(test_img), ef(test_img)))
            assert np.allclose(original, target)
        print("Checkerboard test passed")


if __name__ == "__main__":
    main()