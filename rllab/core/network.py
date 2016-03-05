import lasagne.layers as L
from pydoc import locate


class MLP(object):

    def __init__(self, input_shape, output_dim, hidden_sizes, nonlinearity,
                 output_nl, name=None, input_var=None):
        if isinstance(nonlinearity, basestring):
            nonlinearity = locate(nonlinearity)
        if isinstance(output_nl, basestring):
            output_nl = locate(output_nl)

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        l_hid = l_in
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=nonlinearity,
                name="%shidden_%d" % (prefix, idx)
            )
        l_out = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nl,
            name="%soutput" % (prefix)
        )
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var

    @property
    def l_in(self):
        return self._l_in

    @property
    def l_out(self):
        return self._l_out

    @property
    def input_var(self):
        return self._input_var
