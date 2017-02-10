'''
code originally from khaotik at https://github.com/khaotik/char-rnn-tensorflow

modified by leavesbreathe

Unitary RNN
Reference http://arxiv.org/abs/1511.06464
'''
import numpy as np

import tensorflow as tf

from sandbox.rocky.tf.core.rnn_cells.linear_modern import linear

rnn_cell = tf.nn.rnn_cell

from .complex_util import modrelu_c


def ulinear_c(vec_in_c, scope=None, transform='fourier'):
    '''
    Multiply complex vector by parameterized unitary matrix.
    Equation: W = D2 R1 IT D1 Perm R0 FT D0
    '''
    if not vec_in_c.dtype.is_complex:
        raise ValueError('Argument vec_in_c must be complex valued.')
    shape = vec_in_c.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError('Argument vec_in_c must be a batch of vectors (2D tensor).')
    if transform == 'fourier':
        fwd_trans = tf.batch_fft
        inv_trans = tf.batch_ifft
    elif transform == 'hadamard':
        fwd_trans = batch_fht
        inv_trans = batch_fht
    in_size = shape[1]
    with tf.variable_scope(scope or 'ULinear') as _s:
        diag = [get_unit_variable_c('diag' + i, _s, [in_size]) for i in '012']
        refl = [
            normalize_c(get_variable_c('refl' + i, [in_size], initializer=tf.random_uniform_initializer(-1., 1.))) for i
            in '01']
        perm0 = tf.constant(np.random.permutation(in_size), name='perm0', dtype='int32')
        out = vec_in_c * diag[0]
        out = refl_c(fwd_trans(out), refl[0])
        out = diag[1] * tf.transpose(tf.gather(tf.transpose(out), perm0))
        out = diag[2] * refl_c(inv_trans(out), refl[1])
        if transform == 'fourier':
            return out
        elif transform == 'hadamard':
            return out * (1. / in_size)


# fast hadamard transform, alternative to FFT
# TODO: As of TF version 0.8, this is super slow. Aim for a native cuda implementation
def batch_fht(input):
    def log2n(x):
        i = 0
        while True:
            if x & 1:
                return i if x == 1 else -1
            x >>= 1
            i += 1

    in_shape = input.get_shape().as_list()
    lg2size = log2n(in_shape[-1])
    if lg2size < 0:
        raise (ValueError('fht_c(): The last dimension of input must be power of 2'))
    elif lg2size == 0:
        return input

    idx = [slice(0, i) for i in in_shape[:-1]]
    output = input
    for i in range(lg2size):
        l, r = 2 ** (lg2size - i - 1), 2 ** i
        mid_shape = in_shape[:-1] + [l, 2, r]
        output = tf.reshape(output, mid_shape)
        idx_u = idx + [slice(0, l), slice(0, 1), slice(0, r)]
        idx_v = idx + [slice(0, l), slice(1, 2), slice(0, r)]
        u, v = output[tuple(idx_u)], output[tuple(idx_v)]
        output = tf.concat(len(mid_shape) - 2, [u + v, u - v])
    return tf.reshape(output, in_shape)


class UnitaryRNNCell(rnn_cell.RNNCell):
    def __init__(self, num_units, input_size=None, transform='fourier'):
        self._num_units = num_units
        self._input_size = num_units if input_size == None else input_size
        if transform not in ['fourier', 'hadamard']:
            raise ValueError('URNNCell must use one of following transform: fourier, hadamard')
        self.transform = transform

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        zero_initer = tf.constant_initializer(0.)
        with tf.variable_scope(scope or type(self).__name__):
            # nick there are these two matrix multiplications and they are used to convert regular input sizes to complex outputs -- makes sense -- we can further modify this for lstm configurations
            mat_in = tf.get_variable('W_in', [self.input_size, self.state_size * 2])
            mat_out = tf.get_variable('W_out', [self.state_size * 2, self.output_size])

            in_proj = tf.matmul(inputs, mat_in)
            in_proj_c = tf.complex(in_proj[:, :self.state_size], in_proj[:, self.state_size:])
            out_state = modrelu_c(in_proj_c +
                                  ulinear_c(state, transform=self.transform),
                                  tf.get_variable(name='B', dtype=tf.float32, shape=[self.state_size],
                                                  initializer=zero_initer)
                                  )
            out_bias = tf.get_variable(name='B_out', dtype=tf.float32, shape=[self.output_size],
                                       initializer=zero_initer)
            out = tf.matmul(tf.concat(1, [tf.real(out_state), tf.imag(out_state)]), mat_out) + out_bias
        return out, out_state


class UnitaryWrapperCell(rnn_cell.RNNCell):
    '''this cell allows you to input unitary hidden states into an additional cell 'secondary_cell'
    For example you can do 
    Unitary --> LSTM --> output
    Unitary --> GRU --> output

    important: there are two different hidden states: the cell hidden state and the unitary hidden state


    '''

    def __init__(self, num_units, secondary_cell, input_size=None):
        self._num_units = num_units
        self._input_size = num_units if input_size == None else input_size
        self.secondary_cell = secondary_cell
        self.hidden_bias = tf.constant(tf.random_uniform([num_units], minval=-0.01, maxval=0.01))

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            unitary_hidden_state, secondary_cell_hidden_state = tf.split(1, 2, state)

            mat_in = tf.get_variable('mat_in', [self.input_size, self.state_size * 2])
            mat_out = tf.get_variable('mat_out', [self.state_size * 2, self.output_size])
            in_proj = tf.matmul(inputs, mat_in)
            in_proj_c = tf.complex(tf.split(1, 2, in_proj))
            out_state = modrelu_c(in_proj_c +
                                ulinear_c(unitary_hidden_state, self.state_size),
                                tf.get_variable(name='bias', dtype=tf.float32, shape=tf.shape(unitary_hidden_state),
                                                initializer=tf.constant_initalizer(0.)),
                                scope=scope)

        with tf.variable_scope('unitary_output'):
            '''computes data linear, unitary linear and summation -- TODO: should be complex output'''
            unitary_linear_output_real = linear([tf.real(out_state), tf.imag(out_state), inputs], True, 0.0)

        with tf.variable_scope('scale_nonlinearity'):
            modulus = tf.complex_abs(unitary_linear_output_real)
            rescale = tf.maximum(modulus + hidden_bias, 0.) / (modulus + 1e-7)

        # transition to data shortcut connection


        # out_ = tf.matmul(tf.concat(1,[tf.real(out_state), tf.imag(out_state), ] ), mat_out) + out_bias

        # hidden state is complex but output is completely real
        return out_, out_state  # complex
