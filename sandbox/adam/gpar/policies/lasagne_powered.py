from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides
# import lasagne.layers as L
from importlib import import_module


class LasagnePowered(Parameterized):
    def __init__(self, output_layers):
        self.L = import_module('lasagne.layers')
        # import lasagne.layers as L
        self._output_layers = output_layers
        super(LasagnePowered, self).__init__()

    @property
    def output_layers(self):
        return self._output_layers

    @overrides
    def get_params_internal(self, **tags):  # this gives ALL the vars (not the params values)
        return self.L.get_all_params(  # this lasagne function also returns all var below the passed layers
            self.L.concat(self._output_layers),
            **tags
        )
