import sandbox.rocky.tf.core.layers as L

import tensorflow as tf

class GRUNetwork(object):
    def __init__(self, name, input_shape, output_dim, hidden_dim, hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None, input_var=None, input_layer=None):
        with tf.variable_scope(name):
            if input_layer is None:
                l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
            else:
                l_in = input_layer
            l_step_input = L.InputLayer(shape=(None,) + input_shape, name="step_input")
            l_step_prev_hidden = L.InputLayer(shape=(None, hidden_dim), name="step_prev_hidden")
            l_gru = L.TfGRULayer(l_in, num_units=hidden_dim, hidden_nonlinearity=hidden_nonlinearity,
                               hidden_init_trainable=False, name="gru")
            l_gru_flat = L.ReshapeLayer(
                l_gru, shape=(-1, hidden_dim),
                name="gru_flat"
            )
            l_output_flat = L.DenseLayer(
                l_gru_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat"
            )
            l_output = L.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input:
                tf.reshape(flat_output, tf.pack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))),
                shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output"
            )
            l_step_hidden = l_gru.get_step_layer(l_step_input, l_step_prev_hidden, name="step_hidden")
            l_step_output = L.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                W=l_output_flat.W,
                b=l_output_flat.b,
                name="step_output"
            )

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_hidden = l_step_prev_hidden
            self._l_step_hidden = l_step_hidden
            self._l_step_output = l_step_output

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_hidden_layer(self):
        return self._l_step_prev_hidden

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

def test_tf_gru_layer():
    network = GRUNetwork(name="test", input_shape=(10,), output_dim=20, hidden_dim=30)
    network.input_var
    # l_input = L.InputLayer(shape=(None, 10, 10))
    # l_output = L.TfGRULayer(l_input, num_units=30, horizon=10)
    # L.get_output(l_output)


test_tf_gru_layer()
