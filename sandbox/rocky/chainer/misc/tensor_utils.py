import numpy as np


def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate([np.reshape(x, [-1]) for x in tensors])
    else:
        return np.asarray([])


def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = list(map(np.prod, tensor_shapes))
    indices = np.cumsum(tensor_sizes)[:-1]
    return [np.reshape(pair[0], pair[1]) for pair in zip(np.split(flattened, indices), tensor_shapes)]


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


def sliced_fun_sym(f, n_slices):
    def sliced_f(sliced_inputs, non_sliced_inputs=None):
        if non_sliced_inputs is None:
            non_sliced_inputs = []
        if isinstance(non_sliced_inputs, tuple):
            non_sliced_inputs = list(non_sliced_inputs)
        n_paths = len(sliced_inputs[0])
        slice_size = max(1, n_paths // n_slices)
        ret_vals = None
        for start in range(0, n_paths, slice_size):
            inputs_slice = [v[start:start + slice_size] for v in sliced_inputs]
            slice_ret_vals = f(*(inputs_slice + non_sliced_inputs))
            if not isinstance(slice_ret_vals, (tuple, list)):
                slice_ret_vals_as_list = [slice_ret_vals]
            else:
                slice_ret_vals_as_list = slice_ret_vals
            scaled_ret_vals = [
                v * len(inputs_slice[0]) for v in slice_ret_vals_as_list]
            if ret_vals is None:
                ret_vals = scaled_ret_vals
            else:
                ret_vals = [x + y for x, y in zip(ret_vals, scaled_ret_vals)]
        ret_vals = [v / n_paths for v in ret_vals]
        if not isinstance(slice_ret_vals, (tuple, list)):
            ret_vals = ret_vals[0]
        elif isinstance(slice_ret_vals, tuple):
            ret_vals = tuple(ret_vals)
        return ret_vals

    return sliced_f
