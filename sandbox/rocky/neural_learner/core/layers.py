from sandbox.rocky.tf.core.layers import Layer, XavierUniformInitializer, OrthogonalInitializer

import tensorflow as tf
import numpy as np


class FastWeightsRNNLayer(Layer):
    """
    A fast weights RNN
    """

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity,
                 # Number of inner loops, the S parameter in the paper
                 inner_iters=1,
                 # The lambda parameter in the paper
                 decay=0.95,
                 # The eta parameter in the paper
                 fast_learning_rate=0.5,
                 window_size=50,
                 W_x_init=XavierUniformInitializer(),
                 W_h_init=OrthogonalInitializer(),
                 b_init=tf.zeros_initializer,
                 hidden_init=tf.zeros_initializer,
                 hidden_init_trainable=False,
                 **kwargs):

        if nonlinearity is None:
            nonlinearity = tf.identity

        Layer.__init__(self, incoming=incoming, **kwargs)

        input_shape = self.input_shape[2:]

        input_dim = np.prod(input_shape)

        self.W_x = self.add_param(W_x_init, (input_dim, num_units), name="W_x")
        self.W_h = self.add_param(W_h_init, (num_units, num_units), name="W_h")
        self.b = self.add_param(b_init, (num_units,), name="b")

        # Weights for the initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        self.inner_iters = inner_iters
        self.decay = decay
        self.fast_learning_rate = fast_learning_rate

        self.num_units = num_units
        self.nonlinearity = nonlinearity

    def step(self, hprev, x):
        xb_ruc = tf.matmul(x, self.W_x_ruc) + tf.reshape(self.b_ruc, (1, -1))
        h_ruc = tf.matmul(hprev, self.W_h_ruc)
        xb_r, xb_u, xb_c = tf.split(split_dim=1, num_split=3, value=xb_ruc)
        h_r, h_u, h_c = tf.split(split_dim=1, num_split=3, value=h_ruc)
        r = self.gate_nonlinearity(xb_r + h_r)
        u = self.gate_nonlinearity(xb_u + h_u)
        c = self.nonlinearity(xb_c + r * h_c)
        h = (1 - u) * hprev + u * c
        return h

    def get_step_layer(self, l_in, l_prev_hidden, name=None):
        return GRUStepLayer(incomings=[l_in, l_prev_hidden], recurrent_layer=self, name=name)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        input_shape = tf.shape(input)
        n_batches = input_shape[0]
        n_steps = input_shape[1]
        input = tf.reshape(input, tf.pack([n_batches, n_steps, -1]))
        if 'recurrent_state' in kwargs and self in kwargs['recurrent_state']:
            h0s = kwargs['recurrent_state'][self]
        else:
            h0s = tf.tile(
                tf.reshape(self.h0, (1, self.num_units)),
                (n_batches, 1)
            )
        # flatten extra dimensions
        shuffled_input = tf.transpose(input, (1, 0, 2))
        hs = tf.scan(
            self.step,
            elems=shuffled_input,
            initializer=h0s
        )
        shuffled_hs = tf.transpose(hs, (1, 0, 2))
        if 'recurrent_state_output' in kwargs:
            kwargs['recurrent_state_output'][self] = shuffled_hs
        return shuffled_hs


class GRUStepLayer(MergeLayer):
    def __init__(self, incomings, recurrent_layer, **kwargs):
        super(GRUStepLayer, self).__init__(incomings, **kwargs)
        self._gru_layer = recurrent_layer

    def get_params(self, **tags):
        return self._gru_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0][0]
        return n_batch, self._gru_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = tf.shape(x)[0]
        x = tf.reshape(x, tf.pack([n_batch, -1]))
        x.set_shape((None, self.input_shapes[0][1]))
        stepped = self._gru_layer.step(hprev, x)
        stepped.set_shape((None, self._gru_layer.num_units))
        return stepped
