from rllab import spaces
import numpy as np
import tensorflow as tf

def space2variable(space, prepend_shape, name = None):
    """
    Make variable corresponding to a batch of samples from 'space'
    """
    prepend_shape = list(prepend_shape)
    if isinstance(space, spaces.Box):        
        return tf.placeholder(tf.float32, shape=prepend_shape + [space.flat_dim], name=name)
    elif isinstance(space, spaces.Discrete):
        return tf.placeholder(tf.int32, shape=prepend_shape + [space.flat_dim], name=name)
    else:
        raise NotImplementedError

class ZFilter(object):
    def __init__(self):
        self.initialized = False
        self.mean = None
        self.std = None
        self.count = 0
    def __call__(self, x):
        x = x.astype('float32')
        if self.initialized:
            self.mean = (self.mean * self.count + x.mean(axis=0, keepdims=True)) / (self.count + 1)
            self.std = np.sqrt(
                (np.square(self.std) * self.count + np.square(self.mean - x).mean(axis=0, keepdims=True) )
                / (self.count + 1))
        else:
            self.mean = x.mean(axis=0, keepdims=True)
            self.std = x.std(axis=0, keepdims=True) + 1e-4
            self.initialized = True
        self.count += 1
        return np.clip((x - self.mean) / self.std, -10.0, 10.0)

def discounted_future_sum(X, New, discount):
    """
    X: 2d array of floats, time x features
    New: 2d array of bools, indicating when a new episode has started
    """
    Y = np.zeros_like(X)
    T = X.shape[0]
    Y[T-1] = X[T-1]
    for t in xrange(T-2, -1, -1):
        Y[t] = X[t] + discount * Y[t+1] * (1 - New[t+1])
    return Y


