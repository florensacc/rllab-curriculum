import numpy as np
import operator

def flatten_tensors(tensors):
    return np.concatenate(map(lambda x: np.reshape(x, [-1]), tensors))
    
def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = map(lambda shape: reduce(operator.mul, shape), tensor_shapes)
    indices = np.cumsum(tensor_sizes)[:-1]
    return map(lambda pair: np.reshape(pair[0], pair[1]), zip(np.split(flattened, indices), tensor_shapes))

def high_res_normalize(probs):
    return map(lambda x: x / sum(map(float, probs)), map(float, probs))
