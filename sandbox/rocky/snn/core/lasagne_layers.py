from __future__ import print_function
from __future__ import absolute_import

import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano.tensor as TT
from rllab.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.snn.distributions.standard_bernoulli import StandardBernoulli
from sandbox.rocky.snn.distributions.standard_gaussian import StandardGaussian
from sandbox.rocky.snn.distributions.bernoulli import Bernoulli


class IndependentBernoulliLayer(L.Layer):
    """
    A stochastic layer with Bernoulli random variables as its activations, with parameter 1/2.
    """

    def __init__(self, incoming, num_units, **kwargs):
        super(IndependentBernoulliLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.srng = RandomStreams()

    def get_output_for(self, input, **kwargs):
        return self.get_full_output_for(input, **kwargs)[0]

    def get_full_output_for(self, input, **kwargs):
        if "latent_givens" in kwargs and self in kwargs["latent_givens"]:
            ret = kwargs["latent_givens"][self]
        else:
            N = input.shape[0]
            ret = self.srng.binomial(size=(N, self.num_units), p=0.5)
        return ret, dict(
            distribution=StandardBernoulli(self.num_units),
            dist_info=dict(shape_placeholder=TT.zeros_like(ret)),
        )

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class IndependentGaussianLayer(L.Layer):
    """
    A stochastic layer with standard Gaussian random variables as its activations.
    """

    def __init__(self, incoming, num_units, **kwargs):
        super(IndependentGaussianLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.srng = RandomStreams()

    def get_output_for(self, input, **kwargs):
        return self.get_full_output_for(input, **kwargs)[0]

    def get_full_output_for(self, input, **kwargs):
        if "latent_givens" in kwargs and self in kwargs["latent_givens"]:
            ret = kwargs["latent_givens"][self]
        else:
            N = input.shape[0]
            ret = self.srng.normal(size=(N, self.num_units), avg=0.0, std=1.0)
        return ret, dict(
            distribution=StandardGaussian(self.num_units),
            dist_info=dict(shape_placeholder=TT.zeros_like(ret)),
        )

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class BernoulliLayer(L.DenseLayer):
    """
    A stochastic layer with Bernoulli random variables as its activations. It applies a sigmoid nonlinearity to get
    the distribution parameter p: p = sigmoid(Wx+b). We are guaranteed that 0 <= p <= 1. Also initially p will be
    centered around 1/2, which is desirable.
    """

    def __init__(self, incoming, num_units, W=LI.GlorotUniform(),
                 b=LI.Constant(0.), **kwargs):
        super(BernoulliLayer, self).__init__(
            incoming, num_units, W=W, b=b, nonlinearity=LN.sigmoid, **kwargs)
        self.srng = RandomStreams()

    def get_output_for(self, input, **kwargs):
        return self.get_full_output_for(input, **kwargs)[0]

    def get_full_output_for(self, input, **kwargs):
        activation = super(BernoulliLayer, self).get_output_for(input, **kwargs)
        if "latent_givens" in kwargs and self in kwargs["latent_givens"]:
            ret = kwargs["latent_givens"][self]
        else:
            ret = self.srng.binomial(size=activation.shape, p=activation)
        return ret, dict(
            distribution=Bernoulli(self.num_units),
            dist_info=dict(p=activation),
        )


class GaussianLayer(L.DenseLayer):
    """
    A stochastic layer with Gaussian random variables as its activations. It applies a tanh nonlinearity to get
    the mean parameter: mu = tanh(W_mu.dot(x)+b_mu). We are guaranteed that -1 <= mu <= 1. For the standard deviation
    there are two options: one where the value is independent of the input, and one where it is dependent. In both
    cases we parameterize the log of the standard deviation, so that initially std is around 1. For the second case
    we have a linear layer so that log_std = W_std.dot(x)+b_std.
    """

    def __init__(self, incoming, num_units, W=LI.GlorotUniform(),
                 b=LI.Constant(0.), adaptive_std=False, learn_std=False, init_std=1.0,
                 reparameterize=True, **kwargs):
        """
        :param incoming: incoming layer
        :param num_units: number of independent Gaussian outputs
        :param W: initializer for W
        :param b: initializer for b
        :param adaptive_std: whether std is dependent on the input
        :param learn_std: only in effect if adaptive_std is False. Controls whether to learn the std parameter.
        :param init_std: only in effect if adaptive_std is False. Initial value for the std parameter.
        :param whether to reparameterize
        :return:
        """
        if adaptive_std:
            num_linear_units = num_units * 2
        else:
            num_linear_units = num_units
        super(GaussianLayer, self).__init__(
            incoming, num_linear_units, W=W, b=b, nonlinearity=None, **kwargs)
        if not adaptive_std:
            self.log_std = self.add_param(LI.Constant(init_std), shape=(num_units,), name="log_std",
                                          trainable=learn_std)
        self.srng = RandomStreams()
        self.adaptive_std = adaptive_std
        self.reparameterize = reparameterize

    def get_output_for(self, input, **kwargs):
        return self.get_full_output_for(input, **kwargs)[0]

    def get_full_output_for(self, input, **kwargs):
        activation = super(GaussianLayer, self).get_output_for(input, **kwargs)
        if self.adaptive_std:
            mean = LN.tanh(activation[:, :self.num_units])
            log_std = activation[:, self.num_units:]
        else:
            mean = LN.tanh(activation)
            log_std = TT.tile(self.log_std.dimshuffle('x', 0), (mean.shape[0], 1))
        if "latent_givens" in kwargs and self in kwargs["latent_givens"]:
            given_value = kwargs["latent_givens"][self]
            if self.reparameterize:
                if "latent_dist_infos" in kwargs and self in kwargs["latent_dist_infos"]:
                    dist_info = kwargs["latent_dist_infos"][self]
                    epsilon = dist_info["epsilon"]
                    ret = mean + epsilon * TT.exp(log_std)
                else:
                    # Unfortunately occasionally we might still want the distribution information but not the actual
                    # output... Any better way to do this?
                    ret = given_value
            else:
                ret = given_value
        else:
            epsilon = self.srng.normal(size=mean.shape)
            ret = mean + epsilon * TT.exp(log_std)
        if self.reparameterize:
            return ret, dict(
                distribution=StandardGaussian(self.num_units),
                dist_info=dict(shape_placeholder=TT.zeros_like(ret), epsilon=epsilon),
            )
        else:
            return ret, dict(
                distribution=DiagonalGaussian(self.num_units),
                dist_info=dict(mean=mean, log_std=log_std),
            )
