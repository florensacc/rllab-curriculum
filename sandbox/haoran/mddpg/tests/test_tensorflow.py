import unittest

import numpy as np
import tensorflow as tf

from core import tf_util
from misc.tf_test_case import TFTestCase


def create_network(in_size):
    hidden_sizes = (32, 4)
    nonlinearity = tf.nn.relu
    input_ph = tf.placeholder(tf.float32, shape=[None, in_size])
    last_layer = tf_util.mlp(input_ph, in_size, hidden_sizes, nonlinearity)
    return input_ph, last_layer


class TestTensorFlow(TFTestCase):
    def test_copy_values(self):
        in_size = 10
        with tf.name_scope('a') as _:
            in_a, out_a = create_network(in_size)
        with tf.name_scope('b') as _:
            in_b, out_b = create_network(in_size)

        init = tf.initialize_all_variables()
        self.sess.run(init)

        x = np.random.rand(1, in_size)
        feed_a = {in_a: x}
        feed_b = {in_b: x}
        val_a = self.sess.run(out_a, feed_dict=feed_a)
        val_b = self.sess.run(out_b, feed_dict=feed_b)
        self.assertFalse((val_a == val_b).all())

        # Try copying
        a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "a")
        b_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "b")
        assign_ops = [tf.assign(a_vars[i], b_vars[i]) for i in
                      range(len(a_vars))]
        self.sess.run(assign_ops)
        val_a = self.sess.run(out_a, feed_dict=feed_a)
        val_b = self.sess.run(out_b, feed_dict=feed_b)
        self.assertTrue((val_a == val_b).all())

    def test_get_collections(self):
        in_size = 5
        out_size = 10
        input_placeholder = tf.placeholder(tf.float32, [None, in_size])
        scope = 'abc'
        with tf.name_scope(scope) as _:
            _ = tf_util.linear(input_placeholder,
                               in_size,
                               out_size)
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope)
        self.assertEqual(2, len(variables))
        variables = tf.get_collection(tf.GraphKeys.VARIABLES, "nope")
        self.assertEqual(0, len(variables))

    def test_batch_matmul(self):
        batchsize = 5
        dim = 3
        M = np.random.rand(batchsize, dim, dim)
        x = np.random.rand(batchsize, dim)
        x = np.expand_dims(x, axis=1)
        x_shape = x.shape
        M_shape = M.shape
        x_placeholder = tf.placeholder(tf.float32, x_shape)
        M_placeholder = tf.placeholder(tf.float32, M_shape)

        expected = np.zeros((batchsize, 1, dim))
        for i in range(batchsize):
            expected[i] = np.matmul(x[i], M[i])

        actual = self.sess.run(
            tf.batch_matmul(x_placeholder, M_placeholder),
            feed_dict={
                x_placeholder: x,
                M_placeholder: M,
            })
        self.assertNpEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
