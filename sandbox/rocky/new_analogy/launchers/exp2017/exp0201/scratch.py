import keras
import keras.layers as L
from keras.models import Sequential, Model
import tensorflow as tf
import numpy as np

from sandbox.rocky.new_analogy import keras_ext

keras_ext.inject()


def test_causal():
    input = L.Input(shape=(None, 1))
    out = input
    for rate in [1, 2, 4, 8, 16, 32]:
        out = L.CausalAtrousConv1D(
            border_mode='same',
            nb_filter=1,
            filter_length=10,
            activation=None,
            atrous_rate=rate,
            causal=True,
        )(out)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        x = np.arange(1, 101, dtype=np.float32)
        x = x.reshape((1, -1, 1))

        test_idx = 33
        output = sess.run(out, feed_dict={input: x})[0, test_idx, 0]
        for idx in range(test_idx + 1, 100):
            x[0, idx, 0] += 1
            output_1 = sess.run(out, feed_dict={input: x})[0, test_idx, 0]
            x[0, idx, 0] -= 1
            # changing future value should not affect past values
            np.testing.assert_allclose(output, output_1)
        for idx in range(test_idx + 1):
            x[0, idx, 0] += 1
            output_2 = sess.run(out, feed_dict={input: x})[0, test_idx, 0]
            x[0, idx, 0] -= 1
            # should have some influence
            assert abs(output - output_2) > 1e-7


if __name__ == "__main__":
    test_causal()
