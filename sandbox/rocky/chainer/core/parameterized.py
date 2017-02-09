from contextlib import contextmanager

from rllab.core.serializable import Serializable
from rllab.misc.ext import AttrDict
from sandbox.rocky.chainer.misc import tensor_utils
import numpy as np

load_params = True


@contextmanager
def suppress_params_loading():
    global load_params
    load_params = False
    yield
    load_params = True


class Parameterized(object):
    def __init__(self):
        self._cached_params = {}
        self._cached_param_dtypes = {}
        self._cached_param_shapes = {}

    def get_params_internal(self, **tags):
        """
        Internal method to be implemented which does not perform caching
        """
        raise NotImplementedError

    def get_params(self, **tags):
        """
        Get the list of parameters, filtered by the provided tags.
        Some common tags include 'regularizable' and 'trainable'
        """
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_params:
            self._cached_params[tag_tuple] = self.get_params_internal(**tags)
        return self._cached_params[tag_tuple]

    def get_param_dtypes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_dtypes:
            params = self.get_params(**tags)
            param_values = [x.data for x in params]
            self._cached_param_dtypes[tag_tuple] = [val.dtype for val in param_values]
        return self._cached_param_dtypes[tag_tuple]

    def get_param_shapes(self, **tags):
        tag_tuple = tuple(sorted(list(tags.items()), key=lambda x: x[0]))
        if tag_tuple not in self._cached_param_shapes:
            params = self.get_params(**tags)
            param_values = [x.data for x in params]
            self._cached_param_shapes[tag_tuple] = [val.shape for val in param_values]
        return self._cached_param_shapes[tag_tuple]

    def get_param_values(self, **tags):
        params = self.get_params(**tags)
        param_values = [x.data for x in params]
        return tensor_utils.flatten_tensors(param_values)

    def get_grad_values(self, **tags):
        params = self.get_params(**tags)
        param_values = [x.grad for x in params]
        return tensor_utils.flatten_tensors(param_values)

    def set_param_values(self, flattened_params, **tags):
        param_values = tensor_utils.unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            param.copydata(np.asarray(value, dtype))

    def set_param_values_from(self, other, **tags):
        for param, other_param in zip(
                self.get_params(**tags),
                other.get_params(**tags)):
            param.copydata(other_param)

    def set_grad_values_from(self, other, **tags):
        for param, other_param in zip(
                self.get_params(**tags),
                other.get_params(**tags)):
            param.cleargrad()
            param.addgrad(other_param)

    def flat_to_params(self, flattened_params, **tags):
        return tensor_utils.unflatten_tensors(flattened_params, self.get_param_shapes(**tags))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        global load_params
        if load_params:
            d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        global load_params
        if load_params:
            self.set_param_values(d["params"])


class JointParameterized(Parameterized):
    def __init__(self, components):
        super(JointParameterized, self).__init__()
        self.components = components

    def get_params_internal(self, **tags):
        params = [param for comp in self.components for param in comp.get_params_internal(**tags)]
        # only return unique parameters
        return sorted(set(params), key=hash)
