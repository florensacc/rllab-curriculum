from sandbox.rocky.neural_learner.policies.categorical_dual_rnn_policy import filter_summary
import tensorflow as tf
import numpy as np


def test_filter_summary():
    summary = np.asarray(
        np.concatenate([
            np.arange(100).reshape((1, 100, 1)),
            np.arange(100, 200).reshape((1, 100, 1)),
        ], axis=0), dtype=np.float32
    )
    summary_var = tf.Variable(summary)

    terminal = np.zeros((2, 100, 1), dtype=np.float32)
    terminal[:, ::10, :] = 1
    terminal_var = tf.Variable(terminal)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        result = filter_summary(summary_var, terminal_var, 1).eval()
        assert np.all(result[0].flat[:10] == 0)
        assert np.all(result[0].flat[20:30] == 20)
        assert np.all(result[1].flat[20:30] == 120)
