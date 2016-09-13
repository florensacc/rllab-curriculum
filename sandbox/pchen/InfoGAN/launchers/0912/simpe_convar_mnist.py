import os
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, init_optim
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset
import numpy as np
import tensorflow as tf
import time

from sandbox.pchen.InfoGAN.infogan.misc.distributions import ConvAR, MeanBernoulli

dataset = ResamplingBinarizedMnistDataset(disable_vali=True)
ar_conv_dist = ConvAR(
    tgt_dist=MeanBernoulli(1),
    shape=(28, 28, 1),
    filter_size=3,
    depth=12,
    nr_channels=4,
    pixel_bias=True,
)

# init
sess = tf.Session()
init_x, _ = dataset.train.next_batch(128)
ar_conv_dist.init_mode()
_ = ar_conv_dist.logli(
    init_x.reshape([128, 28, 28, 1]).astype(np.float32)
)

init = tf.initialize_all_variables()
sess.run(init)
print("init")

# train
bs = 64
ar_conv_dist.train_mode()
input_tensor = tf.placeholder(
    tf.float32,
    [bs,] + list(ar_conv_dist._shape),
)
logli = ar_conv_dist.logli(input_tensor)
loss = -tf.reduce_mean(logli)
for init_opt in init_optim():
    if init_opt is None:
        opt = AdamaxOptimizer()
        trainer = opt.minimize(loss)
    else:
        sess.run(init_opt)

last_t = time.time()
for itr in range(100000):
    x, _ = dataset.train.next_batch(bs)
    x = x.reshape([bs] + list(ar_conv_dist._shape))
    _, cur_loss = sess.run([trainer, loss], {input_tensor: x})
    if itr % 50 == 0:
        print("itr: %s, time: %s"  % (itr, time.time() - last_t))
        print(-cur_loss * 28*28)
        last_t = time.time()

