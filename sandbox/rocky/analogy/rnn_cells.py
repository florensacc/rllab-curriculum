import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.python.util import nest

from sandbox.rocky.tf.misc import tensor_utils


def linear(args, output_size, bias, bias_start=0.0, scope=None, weight_normalization=False):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        if weight_normalization:
            v = tf.get_variable("Matrix_wn/v", [total_arg_size, output_size], dtype=dtype)
            g = tf.get_variable("Matrix_wn/g", [output_size], dtype=dtype, initializer=tf.ones_initializer)
            matrix = v * (tf.reshape(g, (1, -1)) / tf.sqrt(tf.reduce_sum(tf.square(v), 0, keep_dims=True)))
        else:
            matrix = tf.get_variable(
                "Matrix", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


class GRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, num_units, activation=tf.nn.tanh, weight_normalization=False):
        self._num_units = num_units
        self._activation = activation
        self._weight_normalization = weight_normalization

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                # We start with bias of 1.0 to not reset and not update.
                # try:
                r, u = tf.split(1, 2, linear([inputs, state], 2 * self._num_units, True, 1.0,
                                             weight_normalization=self._weight_normalization))
                # except Exception as e:
                #     import ipdb; ipdb.set_trace()
                r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)
            with tf.variable_scope("Candidate"):
                c = self._activation(
                    linear([inputs, r * state], self._num_units, True,
                           weight_normalization=self._weight_normalization))
            new_h = u * state + (1 - u) * c
        return new_h, new_h


class AttentionCell(RNNCell):
    def __init__(self, inner_cell, attention_vec_size, num_heads):
        self.inner_cell = inner_cell
        self.attention_vec_size = attention_vec_size
        self.num_heads = num_heads
        self.attend_var = None
        self.attend_proj_vars = None

    @property
    def output_size(self):
        return self.inner_cell.output_size

    @property
    def state_size(self):
        return self.inner_cell.state_size

    def use_extra_inputs(self, extra_inputs, scope=None):
        assert len(extra_inputs) == 2
        attend_var, valid_var = extra_inputs
        self.attend_var = attend_var
        self.valid_var = valid_var
        self.attend_proj_vars = []
        with tf.variable_scope(scope or type(self).__name__):  # "AttentionCell"
            attend_input_dim = attend_var.get_shape().as_list()[-1]
            for head_idx in range(self.num_heads):
                with tf.variable_scope("Attention_%d" % head_idx):
                    W = tf.get_variable(name="W_attn", shape=[attend_input_dim, self.attention_vec_size])
                    self.attend_proj_vars.append(tensor_utils.fast_temporal_matmul(attend_var, W))

    def __call__(self, inputs, state, scope=None):

        def _valid_softmax(x, valids):
            x = x * valids + (-99999) * (1 - valids)
            return tf.nn.softmax(x)

        with tf.variable_scope(scope or type(self).__name__):  # "AttentionCell"
            linear = tf.nn.rnn_cell._linear

            attns = []

            for head_idx in range(self.num_heads):
                with tf.variable_scope("Attention_%d" % head_idx):
                    attend_proj_var = self.attend_proj_vars[head_idx]
                    y = tf.nn.tanh(
                        tf.expand_dims(linear(state, self.attention_vec_size, bias=True), 1) + \
                        attend_proj_var
                    )
                    v = tf.get_variable(name="v", shape=[self.attention_vec_size])
                    s = tf.reduce_sum(v * y, -1)
                    # Compute the attention weights
                    # Here comes the tricky part: we only want to compute softmax over the valid part
                    a = _valid_softmax(s, self.valid_var)
                    # compute the attention-weighted vector d
                    d = tf.reduce_sum(tf.expand_dims(a, -1) * self.attend_var * tf.expand_dims(self.valid_var, -1), 1)
                    attns.append(d)

            return self.inner_cell(tf.concat(1, [inputs] + attns), state)


class AttentionCell_Multiplicative(AttentionCell):
    """
    Variant of attention cell. Instead of linearly add the contribution of the state and the projection, let the state
    control the vector forming the inner product with the projection.
    """

    def __call__(self, inputs, state, scope=None):
        def _valid_softmax(x, valids):
            x = x * valids + (-99999) * (1 - valids)
            return tf.nn.softmax(x)

        with tf.variable_scope(scope or type(self).__name__):  # "AttentionCell"
            linear = tf.nn.rnn_cell._linear

            attns = []

            for head_idx in range(self.num_heads):
                with tf.variable_scope("Attention_%d" % head_idx):
                    attend_proj_var = self.attend_proj_vars[head_idx]
                    # this would be of shape batch_size * attention_dim
                    v = tf.nn.tanh(linear(state, self.attention_vec_size, bias=True))

                    # attention_proj_var: of shape batch_size * n_steps * attention_dim

                    # now form the inner product
                    s = tf.reduce_sum(tf.expand_dims(v, 1) * attend_proj_var, -1)

                    # Compute the attention weights
                    # Here comes the tricky part: we only want to compute softmax over the valid part
                    a = _valid_softmax(s, self.valid_var)
                    # compute the attention-weighted vector d
                    d = tf.reduce_sum(tf.expand_dims(a, -1) * self.attend_var * tf.expand_dims(self.valid_var, -1), 1)
                    attns.append(d)

            return self.inner_cell(tf.concat(1, [inputs] + attns), state)


class AttentionCell_Monotonic(AttentionCell):
    """
    Variant of attention cell. This implements the scheme described in Graves et al. 2013 paper.

    As a function of the hidden state, we generate the following parameters:
    alpha, beta, and kappa:
    """

    @property
    def state_size(self):
        return self.inner_cell.state_size + self.num_heads

    def __call__(self, inputs, state, scope=None):

        prev_kappas = tf.split(split_dim=1, num_split=self.num_heads, value=state[:, -self.num_heads:])
        raw_state = state[:, :-self.num_heads]

        attend_var_shape = tf.shape(self.attend_var)

        N = attend_var_shape[0]
        T = attend_var_shape[1]

        u = tf.cast(
            tf.tile(
                tf.expand_dims(tf.range(T), 0),
                tf.pack((N, 1))
            ),
            tf.float32
        )

        with tf.variable_scope(scope or type(self).__name__):  # "AttentionCell"
            linear = tf.nn.rnn_cell._linear

            attns = []

            kappas = []

            for head_idx in range(self.num_heads):
                with tf.variable_scope("Attention_%d" % head_idx):
                    attend_proj_var = self.attend_proj_vars[head_idx]

                    alpha_hat, beta_hat, kappa_hat = tf.split(
                        split_dim=1, num_split=3, value=linear(raw_state, 3, bias=True))
                    alpha = tf.exp(alpha_hat)
                    beta = tf.exp(beta_hat)
                    kappa = prev_kappas[head_idx] + tf.exp(kappa_hat)

                    # this should have shape batch_size * n_steps
                    phi = alpha * tf.exp(- beta * tf.square(kappa - u))
                    # now multiply it with the attention vector, which has shape batch_size * n_steps * attention_dim
                    w = tf.reduce_sum(tf.expand_dims(phi, -1) * attend_proj_var * tf.expand_dims(self.valid_var, -1), 1)

                    # w should have shape batch_size * attention_dim
                    attns.append(w)
                    kappas.append(kappa)

            cell_output, next_state = self.inner_cell(tf.concat(1, [inputs] + attns), raw_state)
            full_state = tf.concat(1, [next_state] + kappas)
            return cell_output, full_state
