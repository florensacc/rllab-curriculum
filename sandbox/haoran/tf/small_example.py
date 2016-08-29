import tensorflow as tf

theta = tf.Variable(initial_value=1.)


def fn(x, prev):
    return prev + x * theta

result = tf.scan(fn, [1., 2., 3.])

grad_theta = tf.gradients(result, theta)

tf.gradients(grad_theta, theta)
