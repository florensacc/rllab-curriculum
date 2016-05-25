from __future__ import print_function
from __future__ import absolute_import

from rllab.spaces.discrete import Discrete as TheanoDiscrete
import tensorflow as tf


class Discrete(TheanoDiscrete):
    def new_tensor_variable(self, name, extra_dims):
        if self.n <= 2 ** 8:
            return tf.placeholder(tf.uint8, shape=[None] * extra_dims + [self.flat_dim], name=name)
        elif self.n <= 2 ** 16:
            return tf.placeholder(tf.uint16, shape=[None] * extra_dims + [self.flat_dim], name=name)
        else:
            return tf.placeholder(tf.uint32, shape=[None] * extra_dims + [self.flat_dim], name=name)
