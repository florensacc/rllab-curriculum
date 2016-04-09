import numpy as np
import scipy.optimize
from rllab.misc.ext import unflatten_tensor_variables
import theano
import theano.tensor as TT
from rllab.core.parameterized import Parameterized
from rllab.core.lasagne_powered import LasagnePowered
import lasagne.layers as L
import lasagne.nonlinearities as NL
import operator
import matplotlib.pyplot as plt


def optimize_network(loss, network, maxiter=100):
    grads = theano.grad(loss, network.get_params())
    flat_grad = TT.concatenate(map(TT.flatten, grads))
    flat_params = TT.vector('flat_params')
    unflat_params = unflatten_tensor_variables(
        flat_params, network.get_param_shapes(), network.get_params())
    f_opt = theano.function(
        inputs=[flat_params],
        outputs=theano.clone(
            [loss.astype('float64'), flat_grad.astype('float64')],
            replace=zip(network.get_params(), unflat_params)
        ),
        allow_input_downcast=True
    )
    opt_params = scipy.optimize.fmin_l_bfgs_b(
        func=f_opt, x0=network.get_param_values(), maxiter=maxiter)[0]
    network.set_param_values(opt_params)


class MLP(LasagnePowered, Parameterized):

    def __init__(self, input_shape, output_shape, hidden_sizes):
        l_in = L.InputLayer(shape=(None,) + input_shape)

        l_hid = l_in

        for idx, size in enumerate(hidden_sizes):
            l_hid = L.DenseLayer(
                l_hid,
                num_units=size,
                nonlinearity=NL.tanh,
                name="hid%d" % idx
            )

        output_size = reduce(operator.mul, output_shape, 1)

        l_out = L.DenseLayer(
            l_hid,
            num_units=output_size,
            nonlinearity=None,
            name="out"
        )

        Parameterized.__init__(self)
        LasagnePowered.__init__(self, [l_out])

        self._l_out = l_out
        self._l_in = l_in

        self._input_shape = input_shape
        self._output_shape = output_shape

        self._f_predict = theano.function(
            inputs=[l_in.input_var],
            outputs=L.get_output(l_out).reshape((-1,) + output_shape),
            allow_input_downcast=True,
        )

    def predict_sym(self, xs):
        return L.get_output(
            self._l_out, {self._l_in: xs}
        ).reshape((-1,) + self._output_shape)

    def predict(self, xs):
        return self._f_predict(xs)


def visualize(mean, std, x, y):
    lower_bound = (mean - std).flatten()
    upper_bound = (mean + std).flatten()

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    mean = np.asarray(mean).flatten()

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, label="t", color=(1.0, 0, 0, 0.2))
    plt.scatter(x, mean, label="y", color=(0, 0.7, 0, 0.1))
    plt.fill_between(x, lower_bound, upper_bound,
                     facecolor='yellow', alpha=0.5)
    plt.legend()
    plt.show()
