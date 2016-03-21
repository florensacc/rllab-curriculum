import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import pydoc
import theano.tensor as TT
import theano
from rllab.misc import ext
from rllab.core.lasagne_layers import OpLayer
import numpy as np


class MLP(object):

    def __init__(self, input_shape, output_dim, hidden_sizes, nonlinearity,
                 output_nonlinearity, name=None):

        if name is None:
            prefix = ""
        else:
            prefix = name + "_"

        l_in = L.InputLayer(shape=(None,) + input_shape)
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
            nonlinearity=output_nonlinearity,
            name="%soutput" % (prefix,)
        )
        self._l_in = l_in
        self._l_out = l_out
        self._input_var = l_in.input_var

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out


class GRULayer(L.Layer):

    """
    A gated recurrent unit implements the following update mechanism:
    Reset gate:        r(t) = f_r(x(t) @ W_xr + h(t-1) @ W_hr + b_r)
    Update gate:       u(t) = f_u(x(t) @ W_xu + h(t-1) @ W_hu + b_u)
    Cell gate:         c(t) = f_c(x(t) @ W_xc + r(t) * (h(t-1) @ W_hc) + b_c)
    New hidden state:  h(t) = (1 - u(t)) * h(t-1) + u_t * c(t)
    Note that the reset, update, and cell vectors must have the same dimension as the hidden state
    """
    def __init__(self, incoming, num_units, nonlinearity,
                 gate_nonlinearity=LN.sigmoid, name=None,
                 W_init=LI.HeUniform(), b_init=LI.Constant(0.),
                 hidden_init=LI.Constant(0.), hidden_init_trainable=True):

        if nonlinearity is None:
            nonlinearity = LN.identity

        if gate_nonlinearity is None:
            gate_nonlinearity = LN.identity

        super(GRULayer, self).__init__(incoming, name=name)

        input_shape = self.input_shape[2:]

        input_dim = ext.flatten_shape_dim(input_shape)
        # self._name = name
        # Weights for the initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights for the reset gate
        self.W_xr = self.add_param(W_init, (input_dim, num_units), name="W_xr")
        self.W_hr = self.add_param(W_init, (num_units, num_units), name="W_hr")
        self.b_r = self.add_param(b_init, (num_units,), name="b_r", regularizable=False)
        # Weights for the update gate
        self.W_xu = self.add_param(W_init, (input_dim, num_units), name="W_xu")
        self.W_hu = self.add_param(W_init, (num_units, num_units), name="W_hu")
        self.b_u = self.add_param(b_init, (num_units,), name="b_u", regularizable=False)
        # Weights for the cell gate
        self.W_xc = self.add_param(W_init, (input_dim, num_units), name="W_xc")
        self.W_hc = self.add_param(W_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)
        self.gate_nonlinearity = gate_nonlinearity
        self.num_units = num_units
        self.nonlinearity = nonlinearity

    def step(self, x, hprev):
        r = self.gate_nonlinearity(x.dot(self.W_xr) + hprev.dot(self.W_hr) + self.b_r)
        u = self.gate_nonlinearity(x.dot(self.W_xu) + hprev.dot(self.W_hu) + self.b_u)
        c = self.nonlinearity(x.dot(self.W_xc) + r * (hprev.dot(self.W_hc)) + self.b_c)
        h = (1 - u) * hprev + u * c
        return h

    def get_step_layer(self, l_in, l_prev_hidden):
        return GRUStepLayer(incomings=[l_in, l_prev_hidden], gru_layer=self)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        n_batches = input.shape[0]
        n_steps = input.shape[1]
        input = TT.reshape(input, (n_batches, n_steps, -1))
        h0s = TT.tile(TT.reshape(self.h0, (1, self.num_units)), (n_batches, 1))
        # flatten extra dimensions
        shuffled_input = input.dimshuffle(1, 0, 2)
        hs, _ = theano.scan(fn=self.step, sequences=[shuffled_input], outputs_info=h0s)
        shuffled_hs = hs.dimshuffle(1, 0, 2)
        return shuffled_hs


class GRUStepLayer(L.MergeLayer):

    def __init__(self, incomings, gru_layer, name=None):
        super(GRUStepLayer, self).__init__(incomings, name)
        self._gru_layer = gru_layer

    def get_params(self, **tags):
        return self._gru_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0]
        return n_batch, self._gru_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = x.shape[0]
        x = x.reshape((n_batch, -1))
        return self._gru_layer.step(x, hprev)


class GRUNetwork(object):

    def __init__(self, input_shape, output_dim, hidden_dim, nonlinearity=LN.rectify,
                 output_nonlinearity=None, name=None):
        l_in = L.InputLayer(shape=(None, None) + input_shape)
        l_step_input = L.InputLayer(shape=(None,) + input_shape)
        l_step_prev_hidden = L.InputLayer(shape=(None, hidden_dim))
        l_gru = GRULayer(l_in, num_units=hidden_dim, nonlinearity=nonlinearity, hidden_init_trainable=False)
        l_gru_flat = L.ReshapeLayer(
            l_gru, shape=(-1, hidden_dim)
        )
        l_output_flat = L.DenseLayer(
            l_gru_flat,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
        )
        l_output = OpLayer(
            l_output_flat,
            op=lambda flat_output, l_input:
                flat_output.reshape((l_input.shape[0], l_input.shape[1], -1)),
            shape_op=lambda flat_output_shape, l_input_shape:
                (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
            extras=[l_in]
        )
        l_step_hidden = l_gru.get_step_layer(l_step_input, l_step_prev_hidden)
        l_step_output = L.DenseLayer(
            l_step_hidden,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            W=l_output_flat.W,
            b=l_output_flat.b,
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
