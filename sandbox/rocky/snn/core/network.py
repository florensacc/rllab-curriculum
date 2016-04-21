from __future__ import print_function
from __future__ import absolute_import

import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
from sandbox.rocky.snn.core.lasagne_layers import IndependentGaussianLayer, IndependentBernoulliLayer, \
    BernoulliLayer, GaussianLayer
from rllab.core.lasagne_helpers import get_full_output
import theano.tensor as TT


def _pad_latent_tuple(tpl):
    if len(tpl) == 2:
        return tuple(tpl) + (dict(),)
    return tuple(tpl)


def _get_input_latent_layers(l_in, input_latent_vars):
    if input_latent_vars is None:
        return []
    layers = []
    for dist, num_units, options in map(_pad_latent_tuple, input_latent_vars):
        if dist == 'bernoulli':
            cls = IndependentBernoulliLayer
        elif dist == 'gaussian':
            cls = IndependentGaussianLayer
        else:
            raise NotImplementedError
        layers.append(cls(
            l_in,
            num_units=num_units,
            **options
        ))
    return layers


def _get_latent_layers(l_prev, latent_vars):
    if latent_vars is None:
        return []
    layers = []
    for dist, num_units, options in map(_pad_latent_tuple, latent_vars):
        if dist == 'bernoulli':
            cls = BernoulliLayer
        elif dist == 'gaussian':
            cls = GaussianLayer
        else:
            raise NotImplementedError
        layers.append(cls(
            l_prev,
            num_units=num_units,
            **options
        ))
    return layers


class StochasticMLP(object):
    """
    A stochastic multilayer perceptron. Each layer can consist of deterministic nodes and stochastic nodes of
    different types, whose distributions are conditioned on the activations of the previous layer. There can also be
    stochastic nodes at the input layer, which will follow a standard distribution, unless otherwise specified.
    """

    def __init__(
            self,
            input_shape,
            output_dim,
            hidden_sizes,
            input_latent_vars=None,
            hidden_latent_vars=None,
            hidden_nonlinearity=LN.tanh,
            output_nonlinearity=LN.tanh,
            hidden_W_init=LI.GlorotUniform(),
            hidden_b_init=LI.Constant(0.),
            output_W_init=LI.GlorotUniform(),
            output_b_init=LI.Constant(0.),
            input_var=None,
    ):
        """
        Construct a stochastic MLP.
        :param input_shape: a tuple which specifies the shape of the input.
        :param output_dim: int dimension of the output.
        :param hidden_sizes: a list of integers denoting number of deterministic hidden units for each layer.
        :param input_latent_vars: if specified, it should be a list/tuple of tuples of the form (distribution_str,
        num_units), where distribution can be one of 'bernoulli' and 'gaussian'
        :param hidden_latent_vars: if specified, it should be a list/tuple of lists/tuples of tuples, where each
        top-level list/tuple specifies the latent variable configuration for each hidden layer.
        :param hidden_nonlinearity: Nonlinearity applied at each hidden layer to the deterministic units.
        :param output_nonlinearity: Nonlinearity at the output layer.
        :param hidden_W_init: Lasagne initializer for hidden layer W.
        :param hidden_b_init: Lasagne initializer for hidden layer b.
        :param output_W_init: Lasagne initializer for output layer W.
        :param output_b_init: Lasagne initializer for output layer b.
        :param input_var: input symbolic variable
        :return:
        """

        l_latents = []

        l_in = L.InputLayer(shape=(None,) + input_shape, input_var=input_var)
        input_latent_layers = _get_input_latent_layers(l_in, input_latent_vars)

        l_latents.extend(input_latent_layers)

        if len(input_latent_layers) > 0:
            l_joint_in = L.concat([l_in] + input_latent_layers)
        else:
            l_joint_in = l_in

        l_hid = l_joint_in

        if hidden_latent_vars is None:
            hidden_latent_vars = (None,) * len(hidden_sizes)

        for hidden_size, hidden_latent_var in zip(hidden_sizes, hidden_latent_vars):
            l_new_hid = L.DenseLayer(
                l_hid,
                num_units=hidden_size,
                nonlinearity=hidden_nonlinearity,
                W=hidden_W_init,
                b=hidden_b_init,
            )
            l_hid_latent_layers = _get_latent_layers(l_hid, hidden_latent_var)
            if len(l_hid_latent_layers) > 0:
                l_hid = L.concat([l_new_hid] + l_hid_latent_layers)
            else:
                l_hid = l_new_hid
            l_latents.extend(l_hid_latent_layers)

        l_output = L.DenseLayer(
            l_hid,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            W=output_W_init,
            b=output_b_init
        )

        latent_shapes = L.get_output_shape(l_latents, input_shapes=(1,) + input_shape)
        latent_dtypes = [x.dtype for x in L.get_output(l_latents)]

        self._latent_dims = [x[1] for x in latent_shapes]
        self._latent_dtypes = latent_dtypes
        self._l_in = l_in
        self._l_out = l_output
        self._l_latents = l_latents
        self._input_var = l_in.input_var

    @property
    def input_layer(self):
        return self._l_in

    @property
    def latent_layers(self):
        return list(self._l_latents)

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def latent_dims(self):
        return self._latent_dims

    @property
    def latent_dtypes(self):
        return self._latent_dtypes

    # def get_logli_sym(self, input_var, latent_vars):
    #     """
    #     Get the log likelihood of all latent variables, conditioned on the values of other latent variables and
    #     the input layer.
    #     :param latent_vars: a list of latent variables, appeared in the order of layers as would be returned
    #     by the `latent_layers` property
    #     :return:
    #     """
    #     logli = 0.
    #     assert len(latent_vars) == len(self.latent_layers)
    #     latent_vars_dict = dict(zip(self.latent_layers, latent_vars))
    #     _, extras = get_full_output(self._l_out, {self._l_in: input_var}, latent_givens=latent_vars_dict)
    #     for latent_layer in self.latent_layers:
    #         logli += TT.sum(extras[latent_layer]["logli"], axis=1)
    #     return logli
