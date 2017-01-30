from rllab.core.serializable import Serializable
from .base import Space
import numpy as np
from rllab.misc import ext
import theano


class Crown(Space):
    """
    A Crown in R^3.
    I.e., each coordinate is bounded.
    """

    def __init__(self, radius_low, radius_high, shape=None):
        assert np.isscalar(radius_low) and np.isscalar(radius_high)
        self.radius_low = radius_low
        self.radius_high = radius_high

    def sample(self, center):
        self.center = center
        assert center.shape == (3,)
        angle = np.random.uniform(low=0, high=2*np.pi)
        radius = np.random.uniform(low=self.radius_low, high=self.radius_high)
        return center + radius * np.array([np.cos(angle), np.sin(angle), 0])

    def contains(self, x):
        return x.shape == self.shape and (np.linalg.norm(x - self.center) >= self.radius_low) \
               and (np.linalg.norm(x - self.center) <= self.radius_high)

    @property
    def shape(self):
        return self.center.shape

    @property
    def flat_dim(self):
        return np.prod(self.center.shape)

    @property
    def bounds(self):
        return self.radius_low, self.radius_high

    def flatten(self, x):
        return np.asarray(x).flatten()

    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)

    def flatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))

    def unflatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0],) + self.shape)

    def __repr__(self):
        return "Crown" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Crown) and np.allclose(self.radius_high, other.radius_low) and \
               np.allclose(self.radius_high, other.radius_high) and np.allclose(self.center, other.center)

    def __hash__(self):
        return hash((self.radius_low, self.radius_high))

    def new_tensor_variable(self, name, extra_dims):
        return ext.new_tensor(
            name=name,
            ndim=extra_dims+1,
            dtype=theano.config.floatX
        )