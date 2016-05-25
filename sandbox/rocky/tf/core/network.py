from __future__ import print_function
from __future__ import absolute_import

import sandbox.rocky.tf.core.layers as L
import tensorflow as tf


class MLP(object):
    def __init__(self, name, input_shape, output_dim, hidden_sizes, hidden_nonlinearity,
                 output_nonlinearity, hidden_W_init=L.xavier_init, hidden_b_init=tf.zeros_initializer,
                 output_W_init=L.xavier_init, output_b_init=tf.zeros_initializer,
                 input_var=None, input_layer=None):

        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = L.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    W=hidden_W_init,
                    b=hidden_b_init,
                )
                self._layers.append(l_hid)
            l_out = L.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                W=output_W_init,
                b=output_b_init,
            )
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out
            self._input_var = l_in.input_var
            self._output = L.get_output(l_out)

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
    def layers(self):
        return self._layers

    @property
    def output(self):
        return self._output
