from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import tf_go, AdamaxOptimizer, average_grads
from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

import os
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset, ResamplingBinarizedOmniglotDataset, Cifar10Dataset
import sandbox.pchen.InfoGAN.infogan.misc.imported.nn as nn
import rllab.misc.logger as logger

dataset = Cifar10Dataset(dequantized=True)
flat_dim = dataset.image_dim

noise = Gaussian(flat_dim)
shape = [-1, 16, 16, 12]
shaped_noise = ReshapeFlow(
    noise,
    forward_fn=lambda x: tf.reshape(x, shape),
    backward_fn=lambda x: tf_go(x).reshape([-1, noise.dim]).value,
)

# 4 checkerboard flows
# note: unbalanced receptive growth version
def resnet_blocks_gen(blocks=4, filters=64):
    def go(x):
        chns = int_shape(x)[3]
        x = nn.conv2d(x, filters)
        for _ in range(blocks):
            x = nn.gated_resnet(x)
        temp = nn.conv2d(x, chns*2)
        mu = temp[:,:,:,chns:]
        logstd = tf.tanh(temp[:,:,:,:chns]) # might want learn scaling
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
        return go
    def merge(condition, effect):
        shp = int_shape(condition)
        assert len(shp) == 4
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
                    transpose([0, 2, 3, 1]).
                    value
            )
        else:
            raise NotImplementedError

    return split_gen(id), split_gen((id + 1) % 2), merge
cur = shaped_noise
for i in range(2):
    cf, ef, merge = checkerboard_condition_fn_gen(i, True) # fixme: for now
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

#  then 3 channel-wise shearing (note, early noise is not extracted)
def channel_condition_fn_gen(bin=0):
    id = bin % 2
    def split_gen(bit):
        def go(x):
            shp = int_shape(x)
            assert len(shp) == 4
            assert shp[3] % 2 == 0
            cut = shp[3] // 2
            return x[:,:,:,:cut] if bit == 0 else x[:,:,:,cut:]
        return go
    def merge(condition, effect):
        vs = [condition, effect]
        if id != 0:
            vs = [effect, condition]
        return tf.concat(3, vs)
    return split_gen(id), split_gen((id + 1) % 2), merge
for i in range(6):
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
for i in range(5):
    cf, ef, merge = checkerboard_condition_fn_gen(i, True) # fixme: for now
    cur = ShearingFlow(
        cur,
        nn_builder=resnet_blocks_gen(),
        condition_fn=cf,
        effect_fn=ef,
        combine_fn=merge,
    )

dist = cur

device = "/gpu:0"
# get a large batch to do weight norm init
logger.log("Data init start")
init_batch = (dataset.train.next_batch(512)[0]).reshape([-1, 32, 32, 3])
with tf.device("/cpu:0"):
    init_placeholder = tf.placeholder(tf.float32, shape=init_batch.shape)
    dist.init_mode()
    init_logli = dist.logli_prior(init_placeholder)

# train mode
logger.log("Train graph start")
batch_size = 64

optimizer = AdamaxOptimizer(learning_rate=1e-3)

with tf.device(device):
    dist.train_mode()
    train_placeholder = tf.placeholder(tf.float32, shape=(batch_size,)+dataset.image_shape)
    train_logli = dist.logli_prior(train_placeholder)
    loss = -tf.reduce_mean(train_logli)
    tower_grads = optimizer.compute_gradients(loss)
    tower_grads_lst = [tower_grads]
    trainer = optimizer.apply_gradients(grads_and_vars=average_grads(tower_grads_lst))
    init = tf.initialize_all_variables()

sess = tf.Session()
with sess.as_default():
    sess.run(init, {init_placeholder: init_batch})
    logger.log("Data init finished")
    logprobs = []
    for iter in range(1000000):
        if (iter+1) % 200 == 0:
            logger.log("%s bits/dim" % (
                (np.mean(logprobs)/32/32/3 - np.log(256.))/np.log(2)
            ))
            logprobs = []
        if (iter+1) % 1000 == 0:
            test_logprobs = []
            for _ in range(100):
                batch = dataset.validation.next_batch(batch_size)[0].reshape([-1, 32, 32, 3])
                logprob = sess.run(train_logli, feed_dict={train_placeholder: batch})
                test_logprobs.append(logprob)
            logger.log("TEST %s bits/dim" % (
                (np.mean(test_logprobs)/32/32/3 - np.log(256.))/np.log(2)
            ))

        batch = dataset.train.next_batch(batch_size)[0].reshape([-1, 32, 32, 3])
        logprob, _ = sess.run([train_logli, trainer], feed_dict={train_placeholder: batch})
        if np.any(np.isnan(logprob)):
            print("NaN")
            import ipdb; ipdb.set_trace()
        logprobs.append(logprob)

