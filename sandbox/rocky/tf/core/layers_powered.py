from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.core.parameterized import Parameterized
import sandbox.rocky.tf.core.layers as L


class LayersPowered(Parameterized):

    def __init__(self, output_layers):
        self._output_layers = output_layers
        super(LayersPowered, self).__init__()

    def get_params_internal(self, **tags):
        return L.get_all_params(
            L.concat(self._output_layers, "_tmp"),
            **tags
        )
