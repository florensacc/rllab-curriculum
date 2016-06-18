from __future__ import print_function
from __future__ import absolute_import

# We'd like to compute the gradient w.r.t. a discrete random variable, using the straight-through estimator.
from rllab.misc import ext
import theano
import theano.tensor as TT
import numpy as np
import lasagne.updates
from theano.tensor.opt import register_canonicalize

A = theano.shared(np.random.uniform(low=-1, high=1, size=(100, 50)))
b = theano.shared(np.random.uniform(low=-1, high=1, size=(50,)))

act = TT.nnet.sigmoid(A.dot(b))

bits_var = TT.vector("bits")


class CustomGrad(theano.compile.ViewOp):
    def make_node(self, x, known):
        return theano.gof.Apply(self, [x, known], [x.type()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        z[0] = x

    def grad(self, args, g_outs):
        return [g_outs[0], g_outs[0]]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


custom_grad = CustomGrad()
register_canonicalize(theano.gof.PatternSub((custom_grad, 'x', 'y'), 'x'), name='remove_custom_grad')

loss = TT.mean(TT.square(custom_grad(bits_var, act)))

# we want to get the gradient of loss w.r.t. A and b

grads = theano.grad(loss, wrt=[A, b])  # , known_grads={act: theano.grad(loss, bits_var)})

updates = lasagne.updates.adam(grads, params=[A, b])

train = ext.compile_function(
    inputs=[bits_var],
    outputs=[loss, TT.sum(TT.square(A)) + TT.sum(TT.square(b))],
    updates=updates,
    log_name="train"
)

f_act = ext.compile_function(
    inputs=[],
    outputs=act,
    log_name="f_act"
)

for _ in range(1000):
    result = f_act()
    bits = np.cast['int'](np.random.uniform(low=0, high=1, size=(100,)) < result)
    loss_val, param_norm = train(bits)
    print(loss_val, param_norm)  # train(bits))


# import ipdb;
#
# ipdb.set_trace()


# b =
