def test_fancy_index():
    import numpy as np
    from sandbox.rocky.tf.misc.tensor_utils import fancy_index_sym
    import tensorflow as tf

    rand_ids = np.random.choice(np.arange(100), size=100)
    arr = np.random.uniform(low=-1, high=1, size=(100, 100, 100))

    selected = arr[np.arange(100), rand_ids]

    arr_var = tf.Variable(arr, dtype=tf.float32)
    rand_ids_var = tf.Variable(rand_ids, dtype=tf.int32)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        tf_selected = sess.run(fancy_index_sym(arr_var, [tf.range(tf.shape(arr_var)[0]), rand_ids_var]))

    np.testing.assert_allclose(selected, tf_selected)


if __name__ == "__main__":
    test_fancy_index()
