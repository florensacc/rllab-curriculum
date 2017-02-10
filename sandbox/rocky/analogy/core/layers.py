import sandbox.rocky.tf.core.layers as L
import tensorflow as tf

from sandbox.rocky.analogy.rnn_cells import AttentionCell
from sandbox.rocky.tf.misc import tensor_utils


def dynamic_rnn(
        input, cell, parallel_iterations=32, swap_memory=False, time_major=False, scope=None, cell_scope=None,
        state=None
):
    with tf.variable_scope(scope or "dynamic_rnn") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        input_dim = input.get_shape().as_list()[-1]
        assert input_dim is not None

        input_shape = tf.shape(input)
        batch_size = input_shape[0]
        n_steps = input_shape[1]
        if state is None:
            state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        if not time_major:
            # make sure the first dimension of input is time
            input = tf.transpose(input, [1, 0, 2])

        with tf.name_scope(name="dynamic_rnn", values=[]) as scope:
            base_name = scope

        def _new_ta(dtype, name):
            return tf.TensorArray(dtype=dtype, size=n_steps, name=base_name + name)

        input_ta = _new_ta(dtype=input.dtype, name="input")
        input_ta = input_ta.unpack(input)
        output_ta = _new_ta(dtype=tf.float32, name="output")

        time = tf.constant(0, dtype=tf.int32, name="time")

        def compute(time, output_ta_t, state):
            input_t = input_ta.read(time)
            input_t.set_shape((None, input_dim))
            output, new_state = cell(input_t, state, scope=cell_scope)
            output_ta_t = output_ta_t.write(time, output)
            return time + 1, output_ta_t, new_state

        _, output_final_ta, final_state = tf.while_loop(
            lambda time, _1, _2: time < n_steps,
            compute,
            loop_vars=(time, output_ta, state),
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
        )

        final_outputs = output_final_ta.pack()

        if not time_major:
            final_outputs = tf.transpose(final_outputs, [1, 0, 2])

        final_outputs.set_shape((None, None, cell.output_size))

        return final_outputs, final_state


class TfRNNLayer(L.MergeLayer):
    def __init__(self, incoming, cell, **kwargs):
        if not isinstance(incoming, list):
            incoming = [incoming]
        super().__init__(incomings=incoming, **kwargs)
        self.cell = cell
        self.output_dim = cell.output_size
        self.state_dim = cell.state_size

        input_dim = self.input_shapes[0][-1]
        input_dummy = tf.placeholder(tf.float32, (None, input_dim), "input_dummy")
        state_dummy = tf.placeholder(tf.float32, (None, self.state_dim), "state_dummy")
        extra_inputs = [tf.placeholder(tf.float32, shape) for shape in self.input_shapes[1:]]

        with tf.variable_scope(self.name) as vs:
            with tf.variable_scope(cell.__class__.__name__) as cell_scope:
                if hasattr(self.cell, "use_extra_inputs"):
                    self.cell.use_extra_inputs(extra_inputs, scope=cell_scope)
                self.cell(input_dummy, state_dummy, scope=cell_scope)
                self.scope = vs
                self.cell_scope = cell_scope
                all_vars = [v for v in tf.all_variables() if v.name.startswith(cell_scope.name)]
                trainable_vars = [v for v in tf.trainable_variables() if v.name.startswith(cell_scope.name)]
                vs.reuse_variables()
                cell_scope.reuse_variables()

        for var in trainable_vars:
            self.add_param(spec=var, shape=None, name=None, trainable=True)
        for var in set(all_vars) - set(trainable_vars):
            self.add_param(spec=var, shape=None, name=None, trainable=False)

    def get_step_layer(self, incoming, prev_state_layer, name=None):
        return TfRNNStepLayer(incoming=incoming, prev_state_layer=prev_state_layer, recurrent_layer=self, name=name)

    def get_output_for(self, inputs, **kwargs):
        input, *extra_inputs = inputs
        if "recurrent_state" in kwargs and self in kwargs["recurrent_state"]:
            state = kwargs["recurrent_state"][self]
        else:
            state = None

        if hasattr(self.cell, 'use_extra_inputs'):
            self.cell.use_extra_inputs(extra_inputs, scope=self.cell_scope)

        output, final_state = dynamic_rnn(input, self.cell, scope=self.scope, cell_scope=self.cell_scope, state=state)

        if "recurrent_state_output" in kwargs:
            kwargs["recurrent_state_output"][self] = final_state
        return output

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        assert len(input_shape) == 3
        return input_shape[0], input_shape[1], self.cell.output_size


class TfRNNStepLayer(L.MergeLayer):
    def __init__(self, incoming, prev_state_layer, recurrent_layer, **kwargs):
        if not isinstance(incoming, list):
            incoming = [incoming]
        super().__init__(incomings=incoming + [prev_state_layer], **kwargs)
        self.recurrent_layer = recurrent_layer

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][0], self.recurrent_layer.output_shape[-1]

    def get_output_for(self, inputs, **kwargs):
        input, *extra_inputs, prev_state = inputs
        with tf.variable_scope(self.recurrent_layer.scope):
            if hasattr(self.recurrent_layer.cell, "use_extra_inputs"):
                self.recurrent_layer.cell.use_extra_inputs(extra_inputs, scope=self.recurrent_layer.cell_scope)
            output, next_state = self.recurrent_layer.cell(input, prev_state, scope=self.recurrent_layer.cell_scope)
            if 'recurrent_state_output' in kwargs:
                kwargs['recurrent_state_output'][self] = next_state
            return output


class TemporalReverseLayer(L.MergeLayer):
    def __init__(self, incoming, valid_layer, **kwargs):
        super().__init__(incomings=[incoming, valid_layer], **kwargs)

    def get_output_for(self, inputs, **kwargs):
        seq, valids = inputs
        seq_lengths = tf.cast(tf.reduce_sum(valids, reduction_indices=-1), tf.int64)
        return tf.reverse_sequence(seq, seq_lengths, seq_dim=1, batch_dim=0)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]


class TemporalDenseLayer(L.Layer):
    def __init__(self, incoming, num_units, nonlinearity=None, W=L.XavierUniformInitializer(),
                 b=tf.zeros_initializer,
                 **kwargs):
        super().__init__(incoming, **kwargs)
        self.nonlinearity = tf.identity if nonlinearity is None else nonlinearity

        self.num_units = num_units

        assert len(self.input_shape) == 3

        num_inputs = self.input_shape[-1]

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, input, **kwargs):
        input = tf.cast(input, tf.float32)
        activation = tensor_utils.fast_temporal_matmul(input, self.W)
        if self.b is not None:
            activation = activation + tf.expand_dims(tf.expand_dims(self.b, 0), 0)
        return self.nonlinearity(activation)


class AttentionLayer(TfRNNLayer):
    def __init__(self, incoming, attend_layer, valid_layer, cell, attention_cell_cls, attention_vec_size, num_heads=1,
                 **kwargs):
        attention_cell = attention_cell_cls(cell, attention_vec_size=attention_vec_size, num_heads=num_heads)
        super().__init__(incoming=[incoming, attend_layer, valid_layer], cell=attention_cell, **kwargs)
