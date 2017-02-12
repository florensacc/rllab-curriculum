"""Module for constructing RNN Cells with multiplicative_integration"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
import tensorflow as tf
from .multiplicative_integration_modern import multiplicative_integration
from tensorflow.python.ops.nn import rnn_cell
from . import highway_network_modern
from .linear_modern import linear
from . import normalization_ops_modern as nom
from .normalization_ops_modern import layer_norm

RNNCell = rnn_cell.RNNCell


class GRUCell_MulInt_LayerNorm(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units):
        self._num_units = num_units

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, timestep=0, scope=None):
        """Normal Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"

            with tf.variable_scope("Inputs"):
                inputs_concat = linear([inputs], self._num_units * 2, False, 1.0)

                inputs_concat = layer_norm(inputs_concat, num_variables_in_tensor=2, initial_bias_value=1.0)

            with tf.variable_scope("Hidden_State"):
                hidden_state_concat = linear([state], self._num_units * 2, False)

                hidden_state_concat = layer_norm(hidden_state_concat, num_variables_in_tensor=2)

                r, u = tf.split(1, 2, tf.sigmoid(
                    multiplicative_integration([inputs_concat, hidden_state_concat], 2 * self._num_units, 1.0,
                                               weights_already_calculated=True)))

            with tf.variable_scope("Candidate"):
                with tf.variable_scope('input_portion'):
                    input_portion = layer_norm(linear([inputs], self._num_units, False))
                with tf.variable_scope('reset_portion'):
                    reset_portion = r * layer_norm(linear([state], self._num_units, False))

                c = tf.tanh(multiplicative_integration([input_portion, reset_portion], self._num_units, 0.0,
                                                       weights_already_calculated=True))

            new_h = u * state + (1 - u) * c

        return new_h, new_h


class BasicLSTMCell_MulInt_LayerNorm(RNNCell):
    """LSTM cell that has layer norm and mulint capabilities

    layer norm paper: http://arxiv.org/pdf/1607.06450v1.pdf

    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    Biases of the forget gate are initialized by default to 1 in order to reduce
    the scale of forgetting in the beginning of the training.
    """

    def __init__(self, num_units, forget_bias=1.0):
        self._num_units = num_units
        self._forget_bias = forget_bias

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return 2 * self._num_units

    def __call__(self, inputs, state, timestep=0, scope=None):
        """Long short-term memory cell (LSTM).
        The idea with iteration would be to run different batch norm mean and variance stats on timestep greater than 10
        """
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            h, c = tf.split(1, 2, state)

            '''note that bias is set to 0 because batch norm bias is added later'''
            with tf.variable_scope('inputs_weight_matrix'):
                inputs_concat = linear([inputs], 4 * self._num_units, False)

                inputs_concat = layer_norm(inputs_concat, num_variables_in_tensor=4, scope="inputs_concat_layer_norm")

            with tf.variable_scope('state_weight_matrix'):
                h_concat = linear([h], 4 * self._num_units, False)
                h_concat = layer_norm(h_concat, num_variables_in_tensor=4, scope="h_concat_layer_norm")

            i, j, f, o = tf.split(1, 4,
                                  multiplicative_integration([inputs_concat, h_concat], 4 * self._num_units, 0.0,
                                                             weights_already_calculated=True))

            new_c = c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * tf.tanh(j)

            '''apply layer norm to the hidden state transition'''
            with tf.variable_scope('layer_norm_hidden_state'):
                new_h = tf.tanh(layer_norm(new_c)) * tf.sigmoid(o)

        return new_h, tf.concat(1, [new_h, new_c])  # reversed this


class HighwayRNNCell_MulInt_LayerNorm(RNNCell):
    """Highway RNN Network with multiplicative_integration -- has layer norm partially integrated in

    TODO: integrate layer norm on layer one of highway network. Only do this if layer norm even proves to be effective for additional layers"""

    def __init__(self, num_units, num_highway_layers=3, use_recurrent_dropout=False, recurrent_dropout_factor=0.90,
                 is_training=True, use_inputs_on_each_layer=False):
        self._num_units = num_units
        self.num_highway_layers = num_highway_layers
        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_dropout_factor = recurrent_dropout_factor
        self.is_training = is_training
        self.use_inputs_on_each_layer = use_inputs_on_each_layer

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, timestep=0, scope=None):

        current_state = state
        for highway_layer in range(self.num_highway_layers):
            with tf.variable_scope('highway_factor_' + str(highway_layer)):
                if self.use_inputs_on_each_layer or highway_layer == 0:
                    highway_factor = tf.tanh(multiplicative_integration([inputs, current_state], self._num_units))
                else:
                    highway_factor = tf.tanh(layer_norm(linear([current_state], self._num_units, True)))

            with tf.variable_scope('gate_for_highway_factor_' + str(highway_layer)):
                if self.use_inputs_on_each_layer or highway_layer == 0:
                    gate_for_highway_factor = tf.sigmoid(
                        multiplicative_integration([inputs, current_state], self._num_units, initial_bias_value=-3.0))
                else:
                    gate_for_highway_factor = tf.sigmoid(linear([current_state], self._num_units, True, -3.0))

                gate_for_hidden_factor = 1 - gate_for_highway_factor

                if self.use_recurrent_dropout and self.is_training:
                    highway_factor = tf.nn.dropout(highway_factor, self.recurrent_dropout_factor)

            current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

        return current_state, current_state
