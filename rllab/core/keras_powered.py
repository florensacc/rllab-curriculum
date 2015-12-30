from keras.layers.core import Layer
from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc.tensor_utils import flatten_tensors, unflatten_tensors


class KerasPowered(Parameterized):

    def __init__(self, graph):
        self._graph = graph
        self._params = sorted(graph.params, key=lambda x: x.name)
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

    def get_graph_output(self, input, train=False):
        input_layer = Layer()
        input_layer.input = input
        self._graph.set_previous(input_layer)
        return self._graph.get_output(train=train)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])
