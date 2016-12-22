import itertools
import math
import numpy as np
import tensorflow as tf

WEIGHT_DEFAULT_NAME = "weights"
BIAS_DEFAULT_NAME = "bias"


def weight_variable(
        shape,
        initializer=None,
        reuse_variables=False,
        name=WEIGHT_DEFAULT_NAME,
):
    """
    Return a variable with the given shape.

    :param initializer: TensorFlow initializer
    :param reuse_variables:
    :param name:
    :param shape:
    """
    if initializer is None:
        initializer = tf.random_uniform_initializer(minval=-3e-3,
                                                    maxval=3e-3)
    if reuse_variables:
        var = tf.get_variable(name, shape, initializer=initializer)
    else:
        # TODO(vpong): stop hardcoding float32
        var = tf.Variable(initializer(shape), name=name)
    return var


def bias_variable(
        shape,
        initializer=None,
        reuse_variables=False,
        name=BIAS_DEFAULT_NAME,
):
    """
    Return a bias variable with the given shape.

    :param initializer: TensorFlow initializer
    :param reuse_variables:
    :param name:
    :param shape:
    """
    if initializer is None:
        initializer = tf.constant_initializer(0.)
    return weight_variable(shape,
                           initializer=initializer,
                           reuse_variables=reuse_variables,
                           name=name)


def linear(
        last_layer,
        last_size,
        new_size,
        W_initializer=None,
        b_initializer=None,
        reuse_variables=False,
        W_name=WEIGHT_DEFAULT_NAME,
        bias_name=BIAS_DEFAULT_NAME,
):
    """
    Create a linear layer.

    :param W_initializer:
    :param b_initializer:
    :param reuse_variables:
    :param bias_name: String for the bias variables names
    :param W_name: String for the weight matrix variables names
    :param last_layer: Input tensor
    :param last_size: Size of the input tensor
    :param new_size: Size of the output tensor
    :return:
    """
    W = weight_variable([last_size, new_size],
                        initializer=W_initializer,
                        reuse_variables=reuse_variables,
                        name=W_name)
    b = bias_variable((new_size, ),
                      initializer=b_initializer,
                      reuse_variables=reuse_variables,
                      name=bias_name)
    return tf.matmul(last_layer, W) + tf.expand_dims(b, 0)


def mlp(input_layer,
        input_layer_size,
        hidden_sizes,
        nonlinearity,
        W_initializer=None,
        b_initializer=None,
        reuse_variables=False,
        ):
    """
    Create a multi-layer perceptron with the given hidden sizes. The
    nonlinearity is applied after every hidden layer.

    :param b_initializer:
    :param W_initializer:
    :param reuse_variables:
    :param input_layer: tf.Tensor, input to mlp
    :param input_layer_size: int, size of the input
    :param hidden_sizes: int iterable of the hidden sizes
    :param nonlinearity: the initialization function for the nonlinearity
    :return: Output of MLP.
    :type: tf.Tensor
    """
    last_layer = input_layer
    last_layer_size = input_layer_size
    for layer, hidden_size in enumerate(hidden_sizes):
        with tf.variable_scope('hidden{0}'.format(layer)) as _:
            last_layer = nonlinearity(linear(last_layer,
                                             last_layer_size,
                                             hidden_size,
                                             W_initializer=W_initializer,
                                             b_initializer=b_initializer,
                                             reuse_variables=reuse_variables))
            last_layer_size = hidden_size
    return last_layer


def get_lower_triangle_flat_indices(dim):
    indices = []
    for row in range(dim):
        for col in range(dim):
            if col <= row:
                indices.append(row * dim + col)
    return indices


def get_num_elems_in_lower_triangle_matrix(dim):
    return int(dim * (dim + 1) / 2)


# From https://github.com/locuslab/icnn/blob/master/RL/src/naf_nets_dm.py
def vec2lower_triangle(vec, dim):
    """
    Convert a vector M of size (n * m) into a matrix of shape (n, m)
    [[e^M[0],    0,           0,             ...,    0]
     [M[n-1],    e^M[n],      0,      0,     ...,    0]
     [M[2n-1],   M[2n],       e^M[2n+1], 0,  ...,    0]
     ...
     [M[m(n-1)], M[m(n-1)+1], ...,       M[mn-2], e^M[mn-1]]
    """
    L = tf.reshape(vec, [-1, dim, dim])
    L = tf.batch_matrix_band_part(L, -1, 0) - tf.batch_matrix_diag(
        tf.batch_matrix_diag_part(L)) + \
        tf.batch_matrix_diag(tf.exp(tf.batch_matrix_diag_part(L)))
    return L


def quadratic_multiply(x, A):
    """
    Compute x^T A x
    :param x: [n x m] matrix
    :param A: [n x n] matrix
    :return: x^T A x
    """
    return tf.matmul(
        x,
        tf.matmul(
            A,
            x
        ),
        transpose_a=True,
    )


def he_uniform_initializer():
    """He weight initialization.

    Weights are initialized with a standard deviation of
    :math:`\\sigma = gain \\sqrt{\\frac{1}{fan_{in}}}` [1]_.

    References
    ----------
    .. [1] Kaiming He et al. (2015):
           Delving deep into rectifiers: Surpassing human-level performance on
           imagenet classification. arXiv preprint arXiv:1502.01852.
    """

    def _initializer(shape, **kwargs):
        if len(shape) == 2:
            fan_in = shape[0]
        elif len(shape) > 2:
            fan_in = np.prod(shape[1:])
        delta = np.sqrt(1.0 / fan_in)

        # tf.get_variable puts "partition_info" as another kwargs, which is
        # unfortunately not supported by tf.random_uniform
        acceptable_keys = ["seed","name"]
        acceptable_kwargs = {
            key: kwargs[key]
            for key in kwargs
            if key in acceptable_keys
        }
        return tf.random_uniform(shape, minval=-delta, maxval=delta,
            **acceptable_kwargs)

    return _initializer


def xavier_uniform_initializer(shape, **kwargs):
    if len(shape) == 2:
        n_inputs, n_outputs = shape
    else:
        receptive_field_size = np.prod(shape[:2])
        n_inputs = shape[-2] * receptive_field_size
        n_outputs = shape[-1] * receptive_field_size
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range,
                             **kwargs)
