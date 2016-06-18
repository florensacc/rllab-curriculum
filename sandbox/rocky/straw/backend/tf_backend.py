from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf

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
