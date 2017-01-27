import math
import numpy as np
import tensorflow as tf
from . import linear_modern as linear
from tensorflow.python.framework import ops


def highway(input_, output_size, num_layers=2, bias=-2.0, activation=tf.nn.relu, scope=None,
            use_batch_timesteps=False, use_l2_loss=True, timestep=-1):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y

    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.

    if you initially set the bias to -2, then you achieve a simple pass through layer

    use_batch_timesteps requires input to be 3d input [batch_size x timesteps x input_size] and will return a tensor of the exact same dimensions


    """
    if output_size == 'same': output_size = input_.get_shape()[-1]

    linear_function = linear.batch_timesteps_linear if use_batch_timesteps else linear.linear

    with tf.variable_scope(scope or 'highway_network'):
        output = input_
        for idx in range(num_layers):
            original_input = output

            transform_gate = tf.sigmoid(
                linear_function(original_input, output_size, True, bias, scope='transform_lin_%d' % idx,
                                timestep=timestep))
            proposed_output = activation(
                linear_function(original_input, output_size, True, use_l2_loss=use_l2_loss,
                                scope='proposed_output_lin_%d' % idx, timestep=timestep),
                'activation_output_lin_' + str(idx))

            carry_gate = 1.0 - transform_gate

            output = transform_gate * proposed_output + carry_gate * original_input
    return output


def apply_highway_gate(proposed_output, original_input, bias=-2.0):
    '''will apply a sigmoid transform_gate to any proposed output'''

    transform_gate = tf.sigmoid(
        linear.linear(original_input, proposed_output.get_shape()[1], True, bias, scope='transform_lin'))
    carry_gate = 1.0 - transform_gate

    output = transform_gate * proposed_output + carry_gate * original_input
    return output
