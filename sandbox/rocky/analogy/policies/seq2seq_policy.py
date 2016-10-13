import tensorflow as tf
from tensorflow.python.ops import math_ops
# from tensorflow.python.ops.rnn import _rnn_step
from tensorflow.python.ops.rnn import _infer_state_dtype
from tensorflow.python.ops.rnn_cell import _state_size_with_prefix
from tensorflow.python.util import nest
import prettytensor as pt


# pylint: disable=unused-argument
def _rnn_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, state, call_cell, state_size, skip_conditionals=False):
    """Calculate one step of a dynamic RNN minibatch.

    Returns an (output, state) pair conditioned on the sequence_lengths.
    When skip_conditionals=False, the pseudocode is something like:

    if t >= max_sequence_length:
      return (zero_output, state)
    if t < min_sequence_length:
      return call_cell()

    # Selectively output zeros or output, old state or new state depending
    # on if we've finished calculating each row.
    new_output, new_state = call_cell()
    final_output = np.vstack([
      zero_output if time >= sequence_lengths[r] else new_output_r
      for r, new_output_r in enumerate(new_output)
    ])
    final_state = np.vstack([
      state[r] if time >= sequence_lengths[r] else new_state_r
      for r, new_state_r in enumerate(new_state)
    ])
    return (final_output, final_state)

    Args:
      time: Python int, the current time step
      sequence_length: int32 `Tensor` vector of size [batch_size]
      min_sequence_length: int32 `Tensor` scalar, min of sequence_length
      max_sequence_length: int32 `Tensor` scalar, max of sequence_length
      zero_output: `Tensor` vector of shape [output_size]
      state: Either a single `Tensor` matrix of shape `[batch_size, state_size]`,
        or a list/tuple of such tensors.
      call_cell: lambda returning tuple of (new_output, new_state) where
        new_output is a `Tensor` matrix of shape `[batch_size, output_size]`.
        new_state is a `Tensor` matrix of shape `[batch_size, state_size]`.
      state_size: The `cell.state_size` associated with the state.
      skip_conditionals: Python bool, whether to skip using the conditional
        calculations.  This is useful for `dynamic_rnn`, where the input tensor
        matches `max_sequence_length`, and using conditionals just slows
        everything down.

    Returns:
      A tuple of (`final_output`, `final_state`) as given by the pseudocode above:
        final_output is a `Tensor` matrix of shape [batch_size, output_size]
        final_state is either a single `Tensor` matrix, or a tuple of such
          matrices (matching length and shapes of input `state`).

    Raises:
      ValueError: If the cell returns a state tuple whose length does not match
        that returned by `state_size`.
    """

    # Convert state to a list for ease of use
    flat_state = nest.flatten(state)
    flat_zero_output = nest.flatten(zero_output)

    def _copy_one_through(output, new_output):
        copy_cond = (time >= sequence_length)
        return math_ops.select(copy_cond, output, new_output)

    def _copy_some_through(flat_new_output, flat_new_state):
        # Use broadcasting select to determine which values should get
        # the previous state & zero output, and which values should get
        # a calculated state & output.
        flat_new_output = [
            _copy_one_through(zero_output, new_output)
            for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
        flat_new_state = [
            _copy_one_through(state, new_state)
            for state, new_state in zip(flat_state, flat_new_state)]
        return flat_new_output + flat_new_state

    def _maybe_copy_some_through():
        """Run RNN step.  Pass through either no or some past state."""
        new_output, new_state = call_cell()

        nest.assert_same_structure(state, new_state)

        flat_new_state = nest.flatten(new_state)
        flat_new_output = nest.flatten(new_output)
        return tf.cond(
            # if t < min_seq_len: calculate and return everything
            time < min_sequence_length, lambda: flat_new_output + flat_new_state,
            # else copy some of it through
            lambda: _copy_some_through(flat_new_output, flat_new_state))

    # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
    # but benefits from removing cond() and its gradient.  We should
    # profile with and without this switch here.
    if skip_conditionals:
        # Instead of using conditionals, perform the selective copy at all time
        # steps.  This is faster when max_seq_len is equal to the number of unrolls
        # (which is typical for dynamic_rnn).
        new_output, new_state = call_cell()
        nest.assert_same_structure(state, new_state)
        new_state = nest.flatten(new_state)
        new_output = nest.flatten(new_output)
        final_output_and_state = _copy_some_through(new_output, new_state)
    else:
        empty_update = lambda: flat_zero_output + flat_state
        final_output_and_state = tf.cond(
            # if t >= max_seq_len: copy all state through, output zeros
            time >= max_sequence_length, empty_update,
            # otherwise calculation is required: copy some or all of it through
            _maybe_copy_some_through)

    if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
        raise ValueError("Internal error: state and output were not concatenated "
                         "correctly.")
    final_output = final_output_and_state[:len(flat_zero_output)]
    final_state = final_output_and_state[len(flat_zero_output):]

    for output, flat_output in zip(final_output, flat_zero_output):
        output.set_shape(flat_output.get_shape())
    for substate, flat_substate in zip(final_state, flat_state):
        substate.set_shape(flat_substate.get_shape())

    final_output = nest.pack_sequence_as(
        structure=zero_output, flat_sequence=final_output)
    final_state = nest.pack_sequence_as(
        structure=state, flat_sequence=final_state)

    return final_output, final_state


def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
    """Creates a recurrent neural network specified by RNNCell `cell`.

    The simplest form of RNN network generated is:
    ```py
      state = cell.zero_state(...)
      outputs = []
      for input_ in inputs:
        output, state = cell(input_, state)
        outputs.append(output)
      return (outputs, state)
    ```
    However, a few other options are available:

    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.

    The dynamic calculation performed is, at time t for batch row b,
      (output, state)(b, t) =
        (t >= sequence_length(b))
          ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
          : cell(input(b, t), state(b, t - 1))

    Args:
      cell: An instance of RNNCell.
      inputs: A length T list of inputs, each a `Tensor` of shape
        `[batch_size, input_size]`, or a nested tuple of such elements.
      initial_state: (optional) An initial state for the RNN.
        If `cell.state_size` is an integer, this must be
        a `Tensor` of appropriate type and shape `[batch_size, cell.state_size]`.
        If `cell.state_size` is a tuple, this should be a tuple of
        tensors having shapes `[batch_size, s] for s in cell.state_size`.
      dtype: (optional) The data type for the initial state and expected output.
        Required if initial_state is not provided or RNN state has a heterogeneous
        dtype.
      sequence_length: Specifies the length of each sequence in inputs.
        An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
      scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
      A pair (outputs, state) where:
        - outputs is a length T list of outputs (one for each input), or a nested
          tuple of such elements.
        - state is the final state

    Raises:
      TypeError: If `cell` is not an instance of RNNCell.
      ValueError: If `inputs` is `None` or an empty list, or if the input depth
        (column size) cannot be inferred from inputs via shape inference.
    """

    if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
        raise TypeError("cell must be an instance of RNNCell")
    if not nest.is_sequence(inputs):
        raise TypeError("inputs must be a sequence")
    if not inputs:
        raise ValueError("inputs must not be empty")

    outputs = []
    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        # Obtain the first sequence of the input
        first_input = inputs
        while nest.is_sequence(first_input):
            first_input = first_input[0]

        # Temporarily avoid EmbeddingWrapper and seq2seq badness
        # TODO(lukaszkaiser): remove EmbeddingWrapper
        if first_input.get_shape().ndims != 1:

            input_shape = first_input.get_shape().with_rank_at_least(2)
            fixed_batch_size = input_shape[0]

            flat_inputs = nest.flatten(inputs)
            for flat_input in flat_inputs:
                input_shape = flat_input.get_shape().with_rank_at_least(2)
                batch_size, input_size = input_shape[0], input_shape[1:]
                fixed_batch_size.merge_with(batch_size)
                for i, size in enumerate(input_size):
                    if size.value is None:
                        raise ValueError(
                            "Input size (dimension %d of inputs) must be accessible via "
                            "shape inference, but saw value None." % i)
        else:
            fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

        if fixed_batch_size.value:
            batch_size = fixed_batch_size.value
        else:
            batch_size = tf.shape(first_input)[0]
        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError("If no initial_state is provided, "
                                 "dtype must be specified")
            state = cell.zero_state(batch_size, dtype)

        if sequence_length is not None:  # Prepare variables
            def _create_zero_output(output_size):
                # convert int to TensorShape if necessary
                size = _state_size_with_prefix(output_size, prefix=[batch_size])
                output = tf.zeros(
                    tf.pack(size), _infer_state_dtype(dtype, state))
                shape = _state_size_with_prefix(
                    output_size, prefix=[fixed_batch_size.value])
                output.set_shape(tf.TensorShape(shape))
                return output

            output_size = cell.output_size
            flat_output_size = nest.flatten(output_size)
            flat_zero_output = tuple(
                _create_zero_output(size) for size in flat_output_size)
            zero_output = nest.pack_sequence_as(structure=output_size,
                                                flat_sequence=flat_zero_output)

            sequence_length = math_ops.to_int32(sequence_length)
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)

        for time, input_ in enumerate(inputs):
            if time > 0: varscope.reuse_variables()
            # pylint: disable=cell-var-from-loop
            call_cell = lambda: cell(input_, state)
            # pylint: enable=cell-var-from-loop
            if sequence_length is not None:
                (output, state) = _rnn_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    state=state,
                    call_cell=call_cell,
                    state_size=cell.state_size)
            else:
                (output, state) = call_cell()

            outputs.append(output)

        return (outputs, state)


def rnn_decoder(decoder_inputs, initial_state, cell, loop_function=None,
                scope=None):
    """RNN decoder for the sequence-to-sequence model.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor with shape [batch_size x cell.state_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      loop_function: If not None, this function will be applied to the i-th output
        in order to generate the i+1-st input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing generated outputs.
        state: The state of each cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
          (Note that in some cases, like basic RNN cell or GRU cell, outputs and
           states can be the same. They are different for LSTM cells though.)
    """
    with tf.variable_scope(scope or "rnn_decoder"):
        state = initial_state
        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            outputs.append(output)
            if loop_function is not None:
                prev = output
    return outputs, state


"""
Need to modify this model to my need.
First, the decoder outputs must be conditioned on some context (in my case, the outputs are actions, and the context
is the observation). The same context is present in encoder inputs, and maybe the embedding mechanism can be shared (
though this can be a separate component)
"""


def basic_rnn_seq2seq(
        encoder_inputs, decoder_inputs, cell, dtype=tf.float32, scope=None):
    """Basic RNN sequence-to-sequence model.

    This model first runs an RNN to encode encoder_inputs into a state vector,
    then runs decoder, initialized with the last encoder state, on decoder_inputs.
    Encoder and decoder use the same RNN cell type, but don't share parameters.

    Args:
      encoder_inputs: A list of 2D Tensors [batch_size x input_size].
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell in the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with tf.variable_scope(scope or "basic_rnn_seq2seq"):
        _, enc_state = tf.nn.rnn(cell, encoder_inputs, dtype=dtype)
        return rnn_decoder(decoder_inputs, enc_state, cell)


if __name__ == "__main__":
    batch_size = 50
    input_size = 20
    horizon = 30
    hidden_size = 40
    encoder_inputs = []
    decoder_inputs = []
    for t in range(horizon):
        encoder_inputs.append(
            tf.placeholder(
                dtype=tf.float32, shape=(batch_size, input_size), name="encoder_input_%d" % t
            )
        )
        decoder_inputs.append(
            tf.placeholder(
                dtype=tf.float32, shape=(batch_size, input_size), name="decoder_input_%d" % t
            )
        )

    cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)

    basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)
    from tensorflow.models.rnn.translate import seq2seq_model

"""
analogy_obs, demo_obs, demo_actions: list of tf variables
"""


# def obs_embedding(obs, embedding_dim):
#     return pt.wrap(obs).fully_connected(100).fully_connected(embedding_dim)


def batch_embedding(template, inputs):
    """
    :param template: A prettytensor template.
    :param vars: A list of variables indexed by time.
    :return: Apply the template to the batch-concatenated tensor, and then split back to a sequnce.
    """
    horizon = len(inputs)
    batch_size = tf.shape(inputs[0])[0]
    input_dim = inputs[0].get_shape().as_list()[-1]
    flat_inputs = tf.reshape(tf.pack(inputs), tf.pack([horizon * batch_size, input_dim]))
    flat_outputs = template.construct(input=flat_inputs)
    output_dim = flat_outputs.get_shape().as_list()[-1]
    outputs = tf.reshape(flat_outputs, tf.pack([horizon, batch_size, output_dim]))
    output_list = [outputs[t] for t in range(horizon)]
    return output_list


class Seq2seqPolicy(object):
    def __init__(self, env_spec):
        self.hidden_size = 50
        self.embedding_dim = 50

        self.obs_embedding_template = (pt.template("input").
                                       fully_connected(100).
                                       fully_connected(self.embedding_dim))

        self.encoder_template = (pt.template("input").
                                 gru_cell(num_units=self.hidden_size, state=pt.UnboundVariable('state')))

        self.decoder_template = (pt.template("input").
                                 gru_cell(num_units=self.hidden_size, state=pt.UnboundVariable('state')))

    def action_sym(self, demo_obs_vars, demo_action_vars, analogy_obs_vars, valid_vars):
        # analogy_obs = obs_var
        # demo_obs = state_info_vars["demo_obs"]
        # demo_actions = state_info_vars["demo_actions"]

        cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)

        tf.nn.dynamic_rnn


        with tf.variable_scope("seq2seq"):
            # use them as functional interface
            encoder_inputs = batch_embedding(self.obs_embedding_template, demo_obs_vars)
            # demo_obs_vars).construct()
            decoder_inputs = batch_embedding(self.obs_embedding_template, analogy_obs_vars)

            s = tf.zeros((batch_size, ))

            for inp



            # with tf.variable_scope("encoder")

            import ipdb;
            ipdb.set_trace()
            # analogy_obs_vars).construct()

            _, enc_state = tf.nn.rnn(cell, encoder_inputs, dtype=tf.float32)
            actions = rnn_decoder(decoder_inputs, enc_state, cell)

        return tf.pack(actions)
