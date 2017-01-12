# testing pixelcnn's receptive field

import os
from sandbox.pchen.InfoGAN.infogan.misc.custom_ops import AdamaxOptimizer, init_optim
from sandbox.pchen.InfoGAN.infogan.misc.datasets import MnistDataset, FaceDataset, BinarizedMnistDataset, \
    ResamplingBinarizedMnistDataset
import numpy as np
import tensorflow as tf
import time

from sandbox.pchen.InfoGAN.infogan.misc.distributions import *

bs = 12
img_shp = [32,32,3]
dim = np.prod(img_shp)
data_dist = Gaussian(dim)

pixelcnn = PixelCNN(
    nr_resnets=[
        1
    ],
    nr_filters=10,
    nr_extra_nins=0,
    no_downpass=True,
)

# init
sess = tf.Session()
samples = sess.run(data_dist.sample_prior(bs)).reshape([-1,] + img_shp)
pixelcnn.init_mode()
_ = pixelcnn.logli_prior(
    data_dist.sample_prior(bs)
)

init = tf.initialize_all_variables()
sess.run(init)
print("init")

# train
pixelcnn.train_mode()
x_var = tf.placeholder(tf.float32, shape=[bs,] + img_shp)
spatial_logli = pixelcnn.logli(x_var, {}, spatial=True)

truth = sess.run(spatial_logli, feed_dict={x_var: samples})
field = np.zeros(img_shp[:2], dtype=np.int)

def prRed(prt): return ("\033[91m{}\033[00m" .format(prt))
def prGreen(prt): return ("\033[92m{}\033[00m" .format(prt))

while True:
    tgt = eval(input())
    assert isinstance(tgt, list) and len(tgt) == 2
    j, i = tgt
    for xj in range(j+1):
        for xi in range(img_shp[1] if xj != j else i):
            copy = np.copy(samples)
            copy[:, xj, xi, :] += np.inf
            changed = sess.run(spatial_logli, feed_dict={x_var: copy})
            field[xj, xi] = (
                0. if np.allclose(changed[:, j, i], truth[:, j, i]) else 1.
            )
            print(xj, xi, field[xj, xi])

    field[j, i] = 9.
    np.set_printoptions(threshold=np.inf)
    for row in (field.tolist()):
        print(" ".join([
            prRed(i) if i == 9 else prGreen(i) if i == 1 else str(i) for i in row
        ]))

