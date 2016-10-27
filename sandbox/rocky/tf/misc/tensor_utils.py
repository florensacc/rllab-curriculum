import tensorflow as tf
import numpy as np


def compile_function(inputs, outputs, log_name=None, profile=False):
    def run(*input_vals):
        sess = tf.get_default_session()
        if profile:
            run_metadata = tf.RunMetadata()
            out = sess.run(
                outputs,
                feed_dict=dict(list(zip(inputs, input_vals))),
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata,
            )
            from tensorflow.python.client import timeline
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())
            import sys
            sys.exit()
            return out
        return sess.run(outputs, feed_dict=dict(list(zip(inputs, input_vals))))

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


def tensor_shape(t):
    return tuple(k.value for k in t.get_shape())


def tensor_shapes(ts):
    return [tensor_shape(t) for t in ts]


def new_tensor(name, ndim, dtype):
    return tf.placeholder(dtype=dtype, shape=[None] * ndim, name=name)


def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.get_shape().ndims, arr_like.dtype.base_dtype)


def concat_tensor_list(tensor_list):
    return np.concatenate(tensor_list, axis=0)


def concat_tensor_dict_list(tensor_dict_list):
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = concat_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = concat_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def stack_tensor_list(tensor_list):
    return np.array(tensor_list)
    # tensor_shape = np.array(tensor_list[0]).shape
    # if tensor_shape is tuple():
    #     return np.array(tensor_list)
    # return np.vstack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def split_tensor_dict_list(tensor_dict):
    keys = list(tensor_dict.keys())
    ret = None
    for k in keys:
        vals = tensor_dict[k]
        if isinstance(vals, dict):
            vals = split_tensor_dict_list(vals)
        if ret is None:
            ret = [{k: v} for v in vals]
        else:
            for v, cur_dict in zip(vals, ret):
                cur_dict[k] = v
    return ret


def to_onehot_sym(inds, dim):
    return tf.one_hot(inds, depth=dim, on_value=1, off_value=0)


def pad_tensor(x, max_len):
    return np.concatenate([
        x,
        np.tile(np.zeros_like(x[0]), (max_len - len(x),) + (1,) * np.ndim(x[0]))
    ])


def pad_tensor_n(xs, max_len):
    ret = np.zeros((len(xs), max_len) + xs[0].shape[1:], dtype=xs[0].dtype)
    for idx, x in enumerate(xs):
        ret[idx][:len(x)] = x
    return ret


def pad_tensor_dict(tensor_dict, max_len):
    keys = list(tensor_dict.keys())
    ret = dict()
    for k in keys:
        if isinstance(tensor_dict[k], dict):
            ret[k] = pad_tensor_dict(tensor_dict[k], max_len)
        else:
            ret[k] = pad_tensor(tensor_dict[k], max_len)
    return ret


def temporal_flatten_sym(var):
    """
    Assume var is of shape (batch_size, n_steps, feature_dim). Flatten into
    (batch_size * n_steps, feature_dim)
    """
    assert len(var.get_shape().as_list()) == 3
    out = tf.reshape(var, tf.pack([-1, tf.shape(var)[2]]))
    out.set_shape((None, var.get_shape().as_list()[-1]))
    return out


def temporal_unflatten_sym(var, ref_var):
    """
    Assume var is of shape (batch_size * n_steps, feature_dim) and ref_var is of shape (batch_size, n_steps, sth_else).
    Reshape var into shape (batch_size, n_steps, feature_dim)
    """
    var_shape = var.get_shape().as_list()
    ref_var_shape = ref_var.get_shape().as_list()
    out = tf.reshape(
        var,
        tf.pack([tf.shape(ref_var)[0], tf.shape(ref_var)[1], tf.shape(var)[1]])
    )
    out.set_shape((ref_var_shape[0], ref_var_shape[1], var_shape[1]))
    return out


def temporal_tile_sym(var, ref_var):
    """
    Assume var is of shape (batch_size, feature_dim), and ref_var is of shape (batch_size, n_steps, sth_else).
    Tile var along the temporal dimension so that it has new shape (batch_size, n_steps, feature_dim)
    """
    T = tf.shape(ref_var)[1]
    out = tf.tile(
        tf.expand_dims(var, 1),
        tf.pack([1, T, 1]),
    )
    var_shape = var.get_shape().as_list()
    ref_var_shape = ref_var.get_shape().as_list()
    out.set_shape((ref_var_shape[0], ref_var_shape[1], var_shape[1]))
    return out


def fancy_index_sym(x, ids):
    # mechanism: first flatten x into the appropriate shape, compute the ids on this flattened tensor,
    # and then reshape back
    N = tf.shape(ids[0])[0]
    n_ids = len(ids)
    x_shape = tf.shape(x)
    # as list
    x_shape = [x_shape[i] for i in range(len(x.get_shape().dims))]
    flat_x = tf.reshape(x, tf.pack([-1] + x_shape[n_ids:]))
    id_incs = []
    id_inc = 1
    for dim in x_shape[:n_ids][::-1]:
        id_incs.append(id_inc)
        id_inc *= dim
    id_incs = id_incs[::-1]
    flat_ids = tf.zeros((N,), dtype=tf.int32)
    for id, id_inc in zip(ids, id_incs):
        flat_ids += id * id_inc
    selected = tf.gather(flat_x, flat_ids)
    selected.set_shape(ids[0].get_shape().as_list() + x.get_shape().as_list()[n_ids:])
    return selected


def fast_temporal_matmul(x, W):
    """
    Assume that x is of shape (batch_size, n_steps, input_dim) and W is of shape (input_dim, output_dim)
    Compute x.dot(W) which is of shape (batch_size, n_steps, output_dim)
    Utilize 1x1 convolutions, which is found to be slightly faster than directly performing matmul
    """
    x_reshaped = tf.reshape(x, tf.pack([tf.shape(x)[0], tf.shape(x)[1], 1, -1]))
    W_reshaped = tf.reshape(W, tf.pack([1, 1, tf.shape(W)[0], tf.shape(W)[1]]))
    out = tf.reshape(
        tf.nn.conv2d(x_reshaped, W_reshaped, [1, 1, 1, 1], "SAME"),
        tf.pack([tf.shape(x)[0], tf.shape(x)[1], -1])
    )
    x_shape = x.get_shape().as_list()
    W_shape = W.get_shape().as_list()
    out.set_shape((x_shape[0], x_shape[1], W_shape[1]))
    return out


def temporal_matmul(x, W):
    """
    Assume that x is of shape (batch_size, n_steps, input_dim) and W is of shape (input_dim, output_dim)
    Compute x.dot(W) which is of shape (batch_size, n_steps, output_dim)
    This is done by first flattening x into (batch_size * n_steps, input_dim), perform matrix multiplication,
    and then reshape back
    """
    out = tf.reshape(
        tf.matmul(tf.reshape(x, tf.pack([tf.shape(x)[0] * tf.shape(x)[1], -1])), W),
        tf.pack([tf.shape(x)[0], tf.shape(x)[1], -1])
    )
    x_shape = x.get_shape().as_list()
    W_shape = W.get_shape().as_list()
    out.set_shape((x_shape[0], x_shape[1], W_shape[1]))
    return out


def flat_dim(var):
    return int(np.prod([k.value for k in var.get_shape()]))


def single_threaded_session():
    tf_config = tf.ConfigProto(
        device_count={"CPU": 1, "GPU": 1},
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    return tf.Session(config=tf_config)
