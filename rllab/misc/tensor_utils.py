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


def cg(prod, b, x0, tolerance=1.0e-10, max_itr=100):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    conjugate gradient method.

    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method

    ========== Parameters ==========
    prod : a function that computes A*x given x
    b : vector
        The right hand side (RHS) vector of the system.
    x0 : vector
        The starting guess for the solution.
    max_itr : integer
        Maximum number of iterations. Iteration will stop after maxiter 
        steps even if the specified tolerance has not been achieved.
    tolerance : float
        Tolerance to achieve. The algorithm will terminate when either 
        the relative or the absolute residual is below TOLERANCE.
    """

    #   Initializations    
    x = x0
    r0 = b - prod(x)
    p = r0

    #   Start iterations    
    for i in xrange(max_itr):
        a = float(np.dot(r0.T, r0)/np.dot(prod(p).T, p))#np.dot(p.T, A), p))
        x = x + p*a
        ri = r0 - a*prod(p)

        #print i, np.linalg.norm(ri)

        if np.linalg.norm(ri) < tolerance:
            return x
        b = float(np.dot(ri.T, ri)/np.dot(r0.T, r0))
        p = ri + b * p
        r0 = ri
    return x
