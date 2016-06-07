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


def concat_tensor_list(tensor_list):
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    keys = tensor_dict_list[0].keys()
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def to_onehot_sym(inds, dim):
    return tf.one_hot(inds, depth=dim, on_value=1, off_value=0)


def pad_tensor(x, max_len):
    return np.concatenate([
        x,
        np.tile(np.zeros_like(x[0]), (max_len - len(x),) + (1,) * np.ndim(x[0]))
    ])


def pad_tensor_dict(tensor_dict, max_len):
    keys = tensor_dict.keys()
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len)
    return ret
