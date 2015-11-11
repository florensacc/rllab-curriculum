from misc.tensor_utils import flatten_tensors, unflatten_tensors
import lasagne.layers as L
import tensorfuse as theano
from base import Policy
from misc.overrides import overrides


class LasagnePolicy(Policy):

    def __init__(self, output_layers):
        self.params = sorted(L.get_all_params(
            L.concat(output_layers),
            trainable=True
        ), key=lambda x: x.name)
        self.param_shapes = map(
            lambda x: theano.compat.get_value(x, borrow=True).shape,
            self.params
        )
        self.param_dtypes = map(
            lambda x: theano.compat.get_value(x, borrow=True).dtype,
            self.params
        )

    @overrides
    def get_param_values(self):
        return flatten_tensors(map(
            lambda x: theano.compat.get_value(x, borrow=True), self.params
        ))

    @overrides
    def set_param_values(self, flattened_params):
        param_values = unflatten_tensors(flattened_params, self.param_shapes)
        for param, dtype, value in zip(
                self.params,
                self.param_dtypes,
                param_values
                ):
            theano.compat.set_value(param, value.astype(dtype))
