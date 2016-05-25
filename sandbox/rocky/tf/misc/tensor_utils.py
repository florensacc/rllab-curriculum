from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import numpy as np


def compile_function(inputs, outputs, log_name=None):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(zip(inputs, input_vals)))

    return run


def flatten_tensor_variables(ts):
    return tf.concat(0, [tf.reshape(x, [-1]) for x in ts])


def unflatten_tensor_variables(flatarr, shapes, symb_arrs):
    arrs = []
    n = 0
    for (shape, symb_arr) in zip(shapes, symb_arrs):
        size = np.prod(list(shape))
        arr = tf.reshape(flatarr[n:n + size], shape)
        arrs.append(arr)
        n += size
    return arrs


def new_tensor(name, ndim, dtype):
    return tf.placeholder(dtype=dtype, shape=[None] * ndim, name=name)


def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.get_shape().ndims, arr_like.dtype.base_dtype)
