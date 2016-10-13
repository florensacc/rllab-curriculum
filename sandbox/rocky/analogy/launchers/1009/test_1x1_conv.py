# This is to verify whether 1x1 convolutions are faster than direct matrix multiplications
# (as claimed by https://www.reddit.com/r/MachineLearning/comments/3oln72/1x1_convolutions_why_use_them/cvyp4ri,
# and also found in some tf implementations)
# The conclusion is that 1x1 convolution is indeed a bit faster, but not by a large margin
from sandbox.rocky.tf.misc import tensor_utils

if __name__ == "__main__":

    import tensorflow as tf
    import numpy as np

    # initialize a 3-way tensor
    x = np.random.uniform(low=-1, high=1, size=(120, 110, 100))
    W = np.random.uniform(low=-1, high=1, size=(100, 130))

    x_var = tf.Variable(initial_value=x, dtype=tf.float32)
    W_var = tf.Variable(initial_value=W, dtype=tf.float32)

    result_1 = tensor_utils.temporal_matmul(x_var, W_var)
    result_2 = tensor_utils.fast_temporal_matmul(x_var, W_var)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        import time

        # allow both operators to warm start
        sess.run(result_1)
        sess.run(result_2)
        start = time.time()
        for _ in range(100):
            result_1_val = sess.run(result_1)
        end = time.time()
        time1 = end - start
        print("Time 1: %f" % time1)
        start = time.time()
        for _ in range(100):
            result_2_val = sess.run(result_2)
        end = time.time()
        time2 = end - start
        print("Time 2: %f" % time2)
        print("1x1 conv is about %f%% faster than FC" % (100 * (time1 - time2) / time1))
