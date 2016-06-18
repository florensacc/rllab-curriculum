from __future__ import print_function
from __future__ import absolute_import

# We'd like to compute the gradient w.r.t. a discrete random variable, using the straight-through estimator.
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf

import numpy as np

A = tf.Variable(initial_value=np.random.uniform(low=-1, high=1, size=(100, 50)), dtype=tf.float32)
b = tf.Variable(np.random.uniform(low=-1, high=1, size=(50,)), dtype=tf.float32)

act = tf.nn.sigmoid(tf.matmul(A, tf.reshape(b, (50, 1))))[:, 0]

bits_var = tf.placeholder(dtype=tf.float32, shape=[None], name="bits")

custom_py_cnt = 0


def custom_grad(x, y):
    global custom_py_cnt
    custom_py_cnt += 1
    func_name = "CustomPyFunc%d" % custom_py_cnt

    def _func(x, y):
        return x

    @tf.RegisterGradient(func_name)
    def _grad(op, grad):
        return grad, grad

    @tf.RegisterShape(func_name)
    def _shape(op):
        return [op.inputs[0].get_shape(), op.inputs[0].get_shape()]

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": func_name}):
        return tf.py_func(_func, [x, y], [x.dtype])


loss = tf.reduce_mean(tf.square(custom_grad(bits_var, act)))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss, var_list=[A, b])

train = tensor_utils.compile_function(
    inputs=[bits_var],
    outputs=[train_op, loss, tf.reduce_sum(tf.square(A)) + tf.reduce_sum(tf.square(b))],
    log_name="train"
)

f_act = tensor_utils.compile_function(
    inputs=[],
    outputs=act,
    log_name="f_act"
)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for _ in range(1000):
        result = f_act()
        bits = np.cast['int'](np.random.uniform(low=0, high=1, size=(100,)) < result)
        _, loss_val, param_norm = train(bits)
        print(loss_val, param_norm)
