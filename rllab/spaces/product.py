from rllab.spaces.base import Space
import numpy as np
from rllab.misc import ext


class Product(Space):

    def __init__(self, *components):
        if isinstance(components[0], (list, tuple)):
            assert len(components) == 1
            components = components[0]
        self._components = tuple(components)
        dtypes = [c.new_tensor_variable("tmp", extra_dims=0).dtype for c in components]
        self._common_dtype = np.core.numerictypes.find_common_type([], dtypes)

    def sample(self):
        return tuple(x.sample() for x in self._components)

    @property
    def components(self):
        return self._components

    def contains(self, x):
        return all(c.contains(xi) for c, xi in zip(self._components, x))

    def new_tensor_variable(self, name, extra_dims):
        return ext.new_tensor(
            name=name,
            ndim=extra_dims+1,
            dtype=self._common_dtype,
        )

    @property
    def flat_dim(self):
        return np.sum([c.flat_dim for c in self._components])

    def flatten(self, x):
        return np.concatenate([c.flatten(xi) for c, xi in zip(self._components, x)])

    def unflatten(self, x):
        dims = [c.flat_dim for c in self._components]
        flat_xs = np.split(x, np.cumsum(dims)[:-1])
        return tuple(c.unflatten(xi) for c, xi in zip(self._components, flat_xs))

