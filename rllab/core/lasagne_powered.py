from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors
import lasagne.layers as L


class LasagnePowered(Parameterized):

    def __init__(self, output_layers):
        self._params = sorted(L.get_all_params(
            L.concat(output_layers),
            trainable=True
        ), key=lambda x: x.name)
        self._param_shapes = \
            [param.get_value(borrow=True).shape
             for param in self.params]
        self._param_dtypes = \
            [param.get_value(borrow=True).dtype
             for param in self.params]

    @property
    @overrides
    def params(self):
        return self._params

    @property
    @overrides
    def param_dtypes(self):
        return self._param_dtypes

    @property
    @overrides
    def param_shapes(self):
        return self._param_shapes

    @overrides
    def get_param_values(self):
        return flatten_tensors(
            [param.get_value(borrow=True)
             for param in self.params]
        )

    @overrides
    def set_param_values(self, flattened_params):
        param_values = unflatten_tensors(flattened_params, self.param_shapes)
        for param, dtype, value in zip(
                self.params,
                self.param_dtypes,
                param_values):
            param.set_value(value.astype(dtype))

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])

