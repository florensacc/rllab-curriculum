"""Module for constructing RNN Cells"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math, numpy as np
import tensorflow as tf

# from multiplicative_integration import multiplicative_integration, multiplicative_integration_for_multiple_inputs

from tensorflow.python.ops.nn import rnn_cell
from . import highway_network_modern
from .multiplicative_integration_modern import multiplicative_integration
from .normalization_ops_modern import layer_norm

from .linear_modern import linear

RNNCell = rnn_cell.RNNCell


class HighwayRNNCell(RNNCell):
    """Highway RNN Network with multiplicative_integration"""

    def __init__(self, num_units, num_highway_layers=3, use_inputs_on_each_layer=False):
        self._num_units = num_units
        self.num_highway_layers = num_highway_layers
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
                    highway_factor = tf.tanh(linear([inputs, current_state], self._num_units, True))
                else:
                    highway_factor = tf.tanh(linear([current_state], self._num_units, True))
            with tf.variable_scope('gate_for_highway_factor_' + str(highway_layer)):
                if self.use_inputs_on_each_layer or highway_layer == 0:
                    gate_for_highway_factor = tf.sigmoid(linear([inputs, current_state], self._num_units, True, -3.0))
                else:
                    gate_for_highway_factor = tf.sigmoid(linear([current_state], self._num_units, True, -3.0))

                gate_for_hidden_factor = 1.0 - gate_for_highway_factor

            current_state = highway_factor * gate_for_highway_factor + current_state * gate_for_hidden_factor

        return current_state, current_state


class BasicGatedCell(RNNCell):
    """Basic Gated Cell from NasenSpray on reddit: https://www.reddit.com/r/MachineLearning/comments/4vyv89/minimal_gate_unit_for_recurrent_neural_networks/"""

    def __init__(self, num_units, use_multiplicative_integration=True,
                 use_recurrent_dropout=False, recurrent_dropout_factor=0.90, is_training=True,
                 forget_bias_initialization=1.0):
        self._num_units = num_units
        self.use_multiplicative_integration = use_multiplicative_integration
        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_dropout_factor = recurrent_dropout_factor
        self.is_training = is_training
        self.forget_bias_initialization = forget_bias_initialization

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
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates"):  # Forget Gate bias starts as 1.0 -- TODO: double check if this is correct
                if self.use_multiplicative_integration:
                    gated_factor = multiplicative_integration([inputs, state], self._num_units,
                                                              self.forget_bias_initialization)
                else:
                    gated_factor = linear([inputs, state], self._num_units, True, self.forget_bias_initialization)

                gated_factor = tf.sigmoid(gated_factor)

            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([inputs], self._num_units, True, 0.0))

                if self.use_recurrent_dropout and self.is_training:
                    input_contribution = tf.nn.dropout(c, self.recurrent_dropout_factor)
                else:
                    input_contribution = c

            new_h = (1 - gated_factor) * state + gated_factor * input_contribution

        return new_h, new_h


class MGUCell(RNNCell):
    """Minimal Gated Unit from  http://arxiv.org/pdf/1603.09420v1.pdf."""

    def __init__(self, num_units, use_multiplicative_integration=True,
                 use_recurrent_dropout=False, recurrent_dropout_factor=0.90, is_training=True,
                 forget_bias_initialization=1.0):
        self._num_units = num_units
        self.use_multiplicative_integration = use_multiplicative_integration
        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_dropout_factor = recurrent_dropout_factor
        self.is_training = is_training
        self.forget_bias_initialization = forget_bias_initialization

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
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("Gates"):  # Forget Gate bias starts as 1.0 -- TODO: double check if this is correct
                if self.use_multiplicative_integration:
                    gated_factor = multiplicative_integration([inputs, state], self._num_units,
                                                              self.forget_bias_initialization)
                else:
                    gated_factor = linear([inputs, state], self._num_units, True, self.forget_bias_initialization)

                gated_factor = tf.sigmoid(gated_factor)

            with tf.variable_scope("Candidate"):
                if self.use_multiplicative_integration:
                    c = tf.tanh(multiplicative_integration([inputs, state * gated_factor], self._num_units, 0.0))
                else:
                    c = tf.tanh(linear([inputs, state * gated_factor], self._num_units, True, 0.0))

                if self.use_recurrent_dropout and self.is_training:
                    input_contribution = tf.nn.dropout(c, self.recurrent_dropout_factor)
                else:
                    input_contribution = c

            new_h = (1 - gated_factor) * state + gated_factor * input_contribution

        return new_h, new_h


class LSTMCell_MemoryArray(RNNCell):
    """Implementation of Recurrent Memory Array Structures Kamil Rocki
    https://arxiv.org/abs/1607.03085

    Idea is to build more complex memory structures within one single layer rather than stacking multiple layers of RNNs

    """

    def __init__(self, num_units, num_memory_arrays=2, use_multiplicative_integration=True, use_recurrent_dropout=False,
                 recurrent_dropout_factor=0.90, is_training=True, forget_bias=1.0,
                 use_layer_normalization=False):
        self._num_units = num_units
        self.num_memory_arrays = num_memory_arrays
        self.use_multiplicative_integration = use_multiplicative_integration
        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_dropout_factor = recurrent_dropout_factor
        self.is_training = is_training
        self.use_layer_normalization = use_layer_normalization
        self._forget_bias = forget_bias

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units * (self.num_memory_arrays + 1)

    def __call__(self, inputs, state, timestep=0, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            hidden_state_plus_c_list = tf.split(1, self.num_memory_arrays + 1, state)

            h = hidden_state_plus_c_list[0]
            c_list = hidden_state_plus_c_list[1:]

            '''very large matrix multiplication to speed up procedure -- will split variables out later'''

            if self.use_multiplicative_integration:
                concat = multiplicative_integration([inputs, h], self._num_units * 4 * self.num_memory_arrays, 0.0)
            else:
                concat = linear([inputs, h], self._num_units * 4 * self.num_memory_arrays, True)

            if self.use_layer_normalization: concat = layer_norm(concat,
                                                                 num_variables_in_tensor=4 * self.num_memory_arrays)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate -- comes in sets of fours
            all_vars_list = tf.split(1, 4 * self.num_memory_arrays, concat)

            '''memory array loop'''
            new_c_list, new_h_list = [], []
            for array_counter in range(self.num_memory_arrays):

                i = all_vars_list[0 + array_counter * 4]
                j = all_vars_list[1 + array_counter * 4]
                f = all_vars_list[2 + array_counter * 4]
                o = all_vars_list[3 + array_counter * 4]

                if self.use_recurrent_dropout and self.is_training:
                    input_contribution = tf.nn.dropout(tf.tanh(j), self.recurrent_dropout_factor)
                else:
                    input_contribution = tf.tanh(j)

                new_c_list.append(
                    c_list[array_counter] * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) * input_contribution)

                if self.use_layer_normalization:
                    new_c = layer_norm(new_c_list[-1])
                else:
                    new_c = new_c_list[-1]

                new_h_list.append(tf.tanh(new_c) * tf.sigmoid(o))

            '''sum all new_h components -- could instead do a mean -- but investigate that later'''
            new_h = tf.add_n(new_h_list)

        return new_h, tf.concat(1, [new_h] + new_c_list)  # purposely reversed


class JZS1Cell(RNNCell):
    """Mutant 1 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

    def __init__(self, num_units, gpu_for_layer=0, weight_initializer="uniform_unit", orthogonal_scale_factor=1.1):
        self._num_units = num_units
        self._gpu_for_layer = gpu_for_layer
        self._weight_initializer = weight_initializer
        self._orthogonal_scale_factor = orthogonal_scale_factor

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.device("/gpu:" + str(self._gpu_for_layer)):
            """JZS1, mutant 1 with n units cells."""
            with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
                with tf.variable_scope("Zinput"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    '''equation 1 z = sigm(WxzXt+Bz), x_t is inputs'''

                    z = tf.sigmoid(linear([inputs],
                                          self._num_units, True, 1.0, weight_initializer=self._weight_initializer,
                                          orthogonal_scale_factor=self._orthogonal_scale_factor))

                with tf.variable_scope("Rinput"):
                    '''equation 2 r = sigm(WxrXt+Whrht+Br), h_t is the previous state'''

                    r = tf.sigmoid(linear([inputs, state],
                                          self._num_units, True, 1.0, weight_initializer=self._weight_initializer,
                                          orthogonal_scale_factor=self._orthogonal_scale_factor))
                    '''equation 3'''
                with tf.variable_scope("Candidate"):
                    component_0 = linear([r * state],
                                         self._num_units, True)
                    component_1 = tf.tanh(tf.tanh(inputs) + component_0)
                    component_2 = component_1 * z
                    component_3 = state * (1 - z)

                h_t = component_2 + component_3

            return h_t, h_t  # there is only one hidden state output to keep track of.
            # This makes it more mem efficient than LSTM


class JZS2Cell(RNNCell):
    """Mutant 2 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

    def __init__(self, num_units, gpu_for_layer=0, weight_initializer="uniform_unit", orthogonal_scale_factor=1.1):
        self._num_units = num_units
        self._gpu_for_layer = gpu_for_layer
        self._weight_initializer = weight_initializer
        self._orthogonal_scale_factor = orthogonal_scale_factor

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.device("/gpu:" + str(self._gpu_for_layer)):
            """JZS2, mutant 2 with n units cells."""
            with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
                with tf.variable_scope("Zinput"):  # Reset gate and update gate.
                    '''equation 1'''

                    z = tf.sigmoid(linear([inputs, state],
                                          self._num_units, True, 1.0, weight_initializer=self._weight_initializer,
                                          orthogonal_scale_factor=self._orthogonal_scale_factor))

                    '''equation 2 '''
                with tf.variable_scope("Rinput"):
                    r = tf.sigmoid(inputs + (linear([state],
                                                    self._num_units, True, 1.0,
                                                    weight_initializer=self._weight_initializer,
                                                    orthogonal_scale_factor=self._orthogonal_scale_factor)))
                    '''equation 3'''

                with tf.variable_scope("Candidate"):
                    component_0 = linear([state * r, inputs],
                                         self._num_units, True)

                    component_2 = (tf.tanh(component_0)) * z
                    component_3 = state * (1 - z)

                h_t = component_2 + component_3

            return h_t, h_t  # there is only one hidden state output to keep track of.
            # This makes it more mem efficient than LSTM


class JZS3Cell(RNNCell):
    """Mutant 3 of the following paper: http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf"""

    def __init__(self, num_units, gpu_for_layer=0, weight_initializer="uniform_unit", orthogonal_scale_factor=1.1):
        self._num_units = num_units
        self._gpu_for_layer = gpu_for_layer
        self._weight_initializer = weight_initializer
        self._orthogonal_scale_factor = orthogonal_scale_factor

    @property
    def input_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.device("/gpu:" + str(self._gpu_for_layer)):
            """JZS3, mutant 2 with n units cells."""
            with tf.variable_scope(scope or type(self).__name__):  # "JZS1Cell"
                with tf.variable_scope("Zinput"):  # Reset gate and update gate.
                    # We start with bias of 1.0 to not reset and not update.
                    '''equation 1'''

                    z = tf.sigmoid(linear([inputs, tf.tanh(state)],
                                          self._num_units, True, 1.0, weight_initializer=self._weight_initializer,
                                          orthogonal_scale_factor=self._orthogonal_scale_factor))

                    '''equation 2'''
                with tf.variable_scope("Rinput"):
                    r = tf.sigmoid(linear([inputs, state],
                                          self._num_units, True, 1.0, weight_initializer=self._weight_initializer,
                                          orthogonal_scale_factor=self._orthogonal_scale_factor))
                    '''equation 3'''
                with tf.variable_scope("Candidate"):
                    component_0 = linear([state * r, inputs],
                                         self._num_units, True)

                    component_2 = (tf.tanh(component_0)) * z
                    component_3 = state * (1 - z)

                h_t = component_2 + component_3

            return h_t, h_t  # there is only one hidden state output to keep track of.
            # This makes it more mem efficient than LSTM
