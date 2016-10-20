# testing conv af has leakage or not

import os
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, init_optim
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset
import numpy as np
import tensorflow as tf
import time

from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

bs = 128
img_shp = [8,8,4]
dim = np.prod(img_shp)
data_dist = Gaussian(dim)

af_conv_dist = Gaussian(dim)
for _ in range(4):
    af_conv_dist = AR(
        dim,
        af_conv_dist,
        neuron_ratio=2,
        depth=2,
        data_init_wnorm=True,
        img_shape=img_shp,
        data_init_scale=0.01,
        ar_channels=True,
    )

# init
sess = tf.Session()
af_conv_dist.init_mode()
_ = af_conv_dist.logli_prior(
    data_dist.sample_prior(bs)
)

init = tf.initialize_all_variables()
sess.run(init)
print("init")

# train
af_conv_dist.train_mode()
data, data_logli = data_dist.sample_logli(data_dist.prior_dist_info(bs))
logli = af_conv_dist.logli_prior(data)
loss = tf.reduce_mean(
    data_logli - logli
)
for init_opt in init_optim():
    if init_opt is None:
        opt = AdamaxOptimizer(
            learning_rate=0.001
        )
        trainer = opt.minimize(loss)
    else:
        sess.run(init_opt)

eval_every = 50
last_t = time.time()
cur_loss = 0.
for itr in range(100000):
    cur_loss += sess.run([trainer, loss], )[1]
    if itr % eval_every == 0:
        print("itr: %s, time: %s"  % (itr, time.time() - last_t))
        print(cur_loss / eval_every)
        cur_loss = 0.
        last_t = time.time()

