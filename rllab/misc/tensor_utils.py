import numpy as np
import operator


def flatten_tensors(tensors):
    if len(tensors) > 0:
        return np.concatenate(map(lambda x: np.reshape(x, [-1]), tensors))
    else:
        return np.asarray([])

def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = map(lambda shape: reduce(operator.mul, shape), tensor_shapes)
    indices = np.cumsum(tensor_sizes)[:-1]
    return map(lambda pair: np.reshape(pair[0], pair[1]), zip(np.split(flattened, indices), tensor_shapes))


def pad_tensor(x, max_len, rep):
    return np.concatenate([
        x,
        np.tile(rep, (max_len - len(x),) + (1,) * np.ndim(rep))
    ])


def high_res_normalize(probs):
    return map(lambda x: x / sum(map(float, probs)), map(float, probs))


def stack_tensors(tensors):
    tensor_shape = np.array(tensors[0]).shape
    if tensor_shape is tuple():
        return np.array(tensors)
    return np.vstack(tensors)


def stack_tensor_dicts(tensor_dicts):
    keys = tensor_dicts[0].keys()
    return {k: stack_tensors([x[k] for x in tensor_dicts]) for k in keys}


def concat_tensors(tensors):
    return np.concatenate(tensors, axis=0)


def concat_tensor_dicts(tensor_dicts):
    keys = tensor_dicts[0].keys()
    return {k: concat_tensors([x[k] for x in tensor_dicts]) for k in keys}
