from collections import OrderedDict
import theano.tensor as T
from collections import OrderedDict
import theano.tensor as T
import lasagne.layers as L
from sandbox.dave.rllab.core import lasagne_layers as LL


class DilatedVGG(object):
    def __init__(self, l_x, num_encoding_levels=5, channel_inds=None):
        x_shape = l_x.shape[1:]
        assert len(x_shape) == 3

        X_var = T.tensor4('x')
        X_next_var = T.tensor4('x_next')

        l_x_next = L.InputLayer(shape=(None,) + x_shape, input_var=X_next_var, name='x_next')

        xlevels_c_dim = OrderedDict(zip(range(num_encoding_levels+1), [x_shape[0], 64, 128, 256, 512, 512]))

        # encoding
        l_xlevels = OrderedDict()
        for level in range(num_encoding_levels+1):
            if level == 0:
                l_xlevel = l_x
            elif level < 3:
                l_xlevel = LL.DilatedVggEncodingLayer(l_xlevels[level-1], xlevels_c_dim[level], name='conv_dil%d' % level)
            else:
                l_xlevel = LL.DilatedVggEncoding3Layer(l_xlevels[level-1], xlevels_c_dim[level], name='conv_dil%d' % level)
            if level == num_encoding_levels and channel_inds:
                l_xlevel.name += '_all'
                l_xlevel = L.SliceLayer(l_xlevel, channel_inds, axis=1, name='x%d' % level)
            l_xlevels[level] = l_xlevel

        pred_layers = OrderedDict([('x', l_x),
                                   ('x_next', l_x_next),
                                   ])

        self._l_in = l_x
        self._layers = l_xlevels
        self._l_out = l_xlevel
        self._conv_out = l_xlevel

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def conv_output_layer(self):
        return self._conv_out

    @property
    def layers(self):
        return self._layers