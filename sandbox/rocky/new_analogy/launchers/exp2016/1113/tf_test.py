import tensorflow as tf

var = tf.Variable(initial_value=0.)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
