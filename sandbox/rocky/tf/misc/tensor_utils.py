from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf


def compile_function(inputs, outputs):
    def run(*input_vals):
        sess = tf.get_default_session()
        return sess.run(outputs, feed_dict=dict(zip(inputs, input_vals)))
    return run