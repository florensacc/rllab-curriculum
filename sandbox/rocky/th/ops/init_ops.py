import numpy as np


def uniform_initializer(scale):
    scale = abs(scale)

    def _new(shape):
        return np.random.uniform(low=-scale, high=scale, size=shape)

    return _new


def zeros_initializer():
    def _new(shape):
        return np.zeros(shape)

    return _new


def ones_initializer():
    def _new(shape):
        return np.ones(shape)

    return _new
