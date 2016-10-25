from tensorflow.python.ops.rnn_cell import RNNCell
import tensorflow as tf


class EpisodicCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

    def __init__(self, cell):
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size * 2 + self._cell.output_size

    @property
    def output_size(self):
        return self._cell.output_size * 2

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
            input_dim = inputs.get_shape()[-1].value
            assert input_dim is not None
            terminals = tf.cast(inputs[:, input_dim - 1], tf.bool)

            # state has 3 components: trial output, trial state, and episodic state
            s = 0
            prev_trial_out = state[:, :self._cell.output_size]
            s += self._cell.output_size
            prev_trial_state = state[:, s:s + self._cell.state_size]
            s += self._cell.state_size
            prev_episodic_state = state[:, s:s + self._cell.state_size]

            dtype = state.dtype
            batch_size = tf.shape(inputs)[0]

            with tf.variable_scope("Global"):
                # if terminal: update trial state, otherwise stay the same
                trial_out, trial_state = self._cell(prev_episodic_state, prev_trial_state)
                # if not terminal, stick to the previous values
                trial_state = tf.select(terminals, trial_state, prev_trial_state)
                trial_out = tf.select(terminals, trial_out, prev_trial_out)

            with tf.variable_scope("Local"):
                # if terminal: reset prev episodic state
                using_prev_episodic_state = tf.select(
                    terminals,
                    self._cell.zero_state(batch_size, dtype),
                    prev_episodic_state
                )

                episodic_out, episodic_state = self._cell(tf.concat(1, [inputs, trial_out]), using_prev_episodic_state)

            out = tf.concat(1, [trial_out, episodic_out])
            new_state = tf.concat(1, [trial_out, trial_state, episodic_state])

            return out, new_state
