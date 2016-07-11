from __future__ import print_function
import numpy as np
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import ext
from collections import OrderedDict
import theano
from sandbox.rein.dynamics_models.utils import enum

# ----------------
USE_REPARAMETRIZATION_TRICK = True
# ----------------


def conv_input_length(output_length, filter_size, stride, pad=0):
    """Helper function to compute the input size of a convolution operation
    This function computes the length along a single axis, which corresponds
    to a 1D convolution. It can also be used for convolutions with higher
    dimensionalities by using it individually for each axis.
    Parameters
    ----------
    output_length : int or None
        The size of the output.
    filter_size : int
        The size of the filter.
    stride : int
        The stride of the convolution operation.
    pad : int, 'full' or 'same' (default: 0)
        By default, the convolution is only computed where the input and the
        filter fully overlap (a valid convolution). When ``stride=1``, this
        yields an output that is smaller than the input by ``filter_size - 1``.
        The `pad` argument allows you to implicitly pad the input with zeros,
        extending the output size.
        A single integer results in symmetric zero-padding of the given size on
        both borders.
        ``'full'`` pads with one less than the filter size on both sides. This
        is equivalent to computing the convolution wherever the input and the
        filter overlap by at least one position.
        ``'same'`` pads with half the filter size on both sides (one less on
        the second side for an even filter size). When ``stride=1``, this
        results in an output size equal to the input size.
    Returns
    -------
    int or None
        The smallest input size corresponding to the given convolution
        parameters for the given output size, or ``None`` if `output_size` is
        ``None``. For a strided convolution, any input size of up to
        ``stride - 1`` elements larger than returned will still give the same
        output size.
    Raises
    ------
    ValueError
        When an invalid padding is specified, a `ValueError` is raised.
    Notes
    -----
    This can be used to compute the output size of a convolution backward pass,
    also called transposed convolution, fractionally-strided convolution or
    (wrongly) deconvolution in the literature.
    """
    if output_length is None:
        return None
    if pad == 'valid':
        pad = 0
    elif pad == 'full':
        pad = filter_size - 1
    elif pad == 'same':
        pad = filter_size // 2
    if not isinstance(pad, int):
        raise ValueError('Invalid pad: {0}'.format(pad))
    return (output_length - 1) * stride - 2 * pad + filter_size


class ReshapeLayer(lasagne.layers.Layer):
    """
    A layer reshaping its input tensor to another tensor of the same total
    number of elements.
    :parameters:
        - incoming : a :class:`Layer` instance or a tuple
            the layer feeding into this layer, or the expected input shape
        - shape : tuple
            The target shape specification. Any of its elements can be `[i]`,
            a single-element list of int, denoting to use the size of the ith
            input dimension. At most one element can be `-1`, denoting to
            infer the size for this dimension to match the total number of
            elements of the input tensor. Any remaining elements must be
            positive integers directly giving the size of the corresponding
            dimension.
    :usage:
        >>> from lasagne.layers import InputLayer, ReshapeLayer
        >>> l_in = InputLayer((None, 100, 20))
        >>> l1 = ReshapeLayer(l_in, ([0], [1], 2, 10))
        >>> l1.get_output_shape()
        (None, 100, 2, 10)
        >>> l2 = ReshapeLayer(l_in, ([0], 1, 2, 5, -1))
        >>> l2.get_output_shape()
        (None, 1, 2, 5, 200)
    :note:
        The tensor elements will be fetched and placed in C-like order. That
        is, reshaping `[1,2,3,4,5,6]` to shape `(2,3)` will result in a matrix
        `[[1,2,3],[4,5,6]]`, not in `[[1,3,5],[2,4,6]]` (Fortran-like order),
        regardless of the memory layout of the input tensor. For C-contiguous
        input, reshaping is cheap, for others it may require copying the data.
    """

    def __init__(self, incoming, shape, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        shape = tuple(shape)
        for s in shape:
            if isinstance(s, int):
                if s == 0 or s < - 1:
                    raise ValueError("`shape` integers must be positive or -1")
            elif isinstance(s, list):
                if len(s) != 1 or not isinstance(s[0], int) or s[0] < 0:
                    raise ValueError("`shape` input references must be "
                                     "single-element lists of int >= 0")
            else:
                raise ValueError("`shape` must be a tuple of int and/or [int]")
        if sum(s == -1 for s in shape) > 1:
            raise ValueError("`shape` cannot contain multiple -1")
        self.shape = shape

    def get_output_shape_for(self, input_shape, **kwargs):
        # Initialize output shape from shape specification
        output_shape = list(self.shape)
        # First, replace all `[i]` with the corresponding input dimension, and
        # mask parts of the shapes thus becoming irrelevant for -1 inference
        masked_input_shape = list(input_shape)
        masked_output_shape = list(output_shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                if o[0] >= len(input_shape):
                    raise ValueError("specification contains [%d], but input "
                                     "shape has %d dimensions only" %
                                     (o[0], len(input_shape)))
                output_shape[dim] = input_shape[o[0]]
                masked_output_shape[dim] = input_shape[o[0]]
                if (input_shape[o[0]] is None) \
                   and (masked_input_shape[o[0]] is None):
                    # first time we copied this unknown input size: mask
                    # it, we have a 1:1 correspondence between out[dim] and
                    # in[o[0]] and can ignore it for -1 inference even if
                    # it is unknown.
                    masked_input_shape[o[0]] = 1
                    masked_output_shape[dim] = 1
        # From the shapes, compute the sizes of the input and output tensor
        input_size = (None if any(x is None for x in masked_input_shape)
                      else np.prod(masked_input_shape))
        output_size = (None if any(x is None for x in masked_output_shape)
                       else np.prod(masked_output_shape))
        del masked_input_shape, masked_output_shape
        # Finally, infer value for -1 if needed
        if -1 in output_shape:
            dim = output_shape.index(-1)
            if (input_size is None) or (output_size is None):
                output_shape[dim] = None
                output_size = None
            else:
                output_size *= -1
                output_shape[dim] = input_size // output_size
                output_size *= output_shape[dim]
        # Sanity check
        if (input_size is not None) and (output_size is not None) \
           and (input_size != output_size):
            raise ValueError("%s cannot be reshaped to specification %s. "
                             "The total size mismatches." %
                             (input_shape, self.shape))
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        # Replace all `[i]` with the corresponding input dimension
        output_shape = list(self.shape)
        for dim, o in enumerate(output_shape):
            if isinstance(o, list):
                output_shape[dim] = input.shape[o[0]]
        # Everything else is handled by Theano
        return input.reshape(tuple(output_shape))

reshape = ReshapeLayer  # shortcut


class TransConvLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                 crop=0, untie_biases=False,
                 W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,
                 **kwargs):

        super(TransConvLayer, self).__init__(
            incoming, **kwargs)

        pad = crop
        self.crop = crop
        self.n = len(self.input_shape) - 2
        self.nonlinearity = nonlinearity
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, self.n, int)
        self.flip_filters = flip_filters
        self.stride = lasagne.utils.as_tuple(stride, self.n, int)
        self.untie_biases = untie_biases

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = lasagne.utils.as_tuple(0, self.n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = lasagne.utils.as_tuple(pad, self.n, int)

        self.W = self.add_param(W, self.get_W_shape(), name="W")
        if b is None:
            self.b = None
        else:
            if self.untie_biases:
                biases_shape = (num_filters,) + self.output_shape[2:]
            else:
                biases_shape = (num_filters,)
            self.b = self.add_param(b, biases_shape, name="b",
                                    regularizable=False)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) +
                tuple(conv_input_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=self.get_W_shape(),
            subsample=self.stride, border_mode=border_mode,
            filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.b is None:
            activation = conved
        elif self.untie_biases:
            activation = conved + T.shape_padleft(self.b, 1)
        else:
            activation = conved + self.b.dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)


class UpscaleLayer(lasagne.layers.Layer):
    """
    2D upscaling layer
    Performs 2D upscaling over the two trailing axes of a 4D input tensor.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or tuple
        The layer feeding into this layer, or the expected input shape.
    scale_factor : integer or iterable
        The scale factor in each dimension. If an integer, it is promoted to
        a square scale factor region. If an iterable, it should have two
        elements.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    """

    def __init__(self, incoming, scale_factor, **kwargs):
        super(UpscaleLayer, self).__init__(incoming, **kwargs)

        self.scale_factor = lasagne.utils.as_tuple(scale_factor, 2)

        if self.scale_factor[0] < 1 or self.scale_factor[1] < 1:
            raise ValueError('Scale factor must be >= 1, not {0}'.format(
                self.scale_factor))

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list
        if output_shape[2] is not None:
            output_shape[2] *= self.scale_factor[0]
        if output_shape[3] is not None:
            output_shape[3] *= self.scale_factor[1]
        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        a, b = self.scale_factor
        upscaled = input
        if b > 1:
            upscaled = T.extra_ops.repeat(upscaled, b, 3)
        if a > 1:
            upscaled = T.extra_ops.repeat(upscaled, a, 2)
        return upscaled


class PoolLayer(lasagne.layers.Layer):

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(PoolLayer, self).__init__(incoming, **kwargs)

        self.pool_size = lasagne.utils.as_tuple(pool_size, 2)

        if len(self.input_shape) != 4:
            raise ValueError("Tried to create a 2D pooling layer with "
                             "input shape %r. Expected 4 input dimensions "
                             "(batchsize, channels, 2 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = lasagne.utils.as_tuple(stride, 2)

        self.pad = lasagne.utils.as_tuple(pad, 2)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = lasagne.layers.pool.pool_output_length(input_shape[2],
                                                                 pool_size=self.pool_size[
                                                                     0],
                                                                 stride=self.stride[
                                                                     0],
                                                                 pad=self.pad[
                                                                     0],
                                                                 ignore_border=self.ignore_border,
                                                                 )

        output_shape[3] = lasagne.layers.pool.pool_output_length(input_shape[3],
                                                                 pool_size=self.pool_size[
                                                                     1],
                                                                 stride=self.stride[
                                                                     1],
                                                                 pad=self.pad[
                                                                     1],
                                                                 ignore_border=self.ignore_border,
                                                                 )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = theano.tensor.signal.pool.pool_2d(input,
                                                   ds=self.pool_size,
                                                   st=self.stride,
                                                   ignore_border=self.ignore_border,
                                                   padding=self.pad,
                                                   mode=self.mode,
                                                   )
        return pooled

# -----------------------------------------------------------------------


class BayesianLayer(lasagne.layers.Layer):
    """Generic Bayesian layer"""

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 prior_sd=None,
                 disable_variance=False,
                 **kwargs):
        super(BayesianLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        self.nonlinearity = nonlinearity
        self.prior_sd = prior_sd
        self.num_units = num_units
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.prior_rho = self.std_to_log(self.prior_sd)
        self.disable_variance = disable_variance

    def init_params(self):
        self.mu = self.add_param(
            lasagne.init.Normal(1., 0.), self.get_W_shape(), name='mu')

        self.rho = self.add_param(
            lasagne.init.Constant(self.prior_rho), self.get_W_shape(), name='rho')

        self.b_mu = self.add_param(
            lasagne.init.Constant(self.prior_rho), self.get_b_shape(), name="b_mu", regularizable=False)

        self.b_rho = self.add_param(
            lasagne.init.Constant(self.prior_rho), self.get_b_shape(), name="b_rho", regularizable=False)

        # Backup params for KL calculations.
        self.mu_old = self.add_param(
            np.zeros(self.get_W_shape()), self.get_W_shape(), name='mu_old', trainable=False, oldparam=True)

        self.rho_old = self.add_param(
            np.zeros(self.get_W_shape()),  self.get_W_shape(), name='rho_old', trainable=False, oldparam=True)

        # Bias priors.
        self.b_mu_old = self.add_param(
            np.zeros(self.get_b_shape()), self.get_b_shape(),  name="b_mu_old", regularizable=False,   trainable=False, oldparam=True)

        self.b_rho_old = self.add_param(
            np.zeros(self.get_b_shape()), self.get_b_shape(), name="b_rho_old", regularizable=False, trainable=False, oldparam=True)

    def log_to_std(self, rho):
        """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+

        This makes sure that we don't get negative stds. However, a downside might be
        that we have little gradient on close to 0 std (= -inf using this transformation).
        """
        return T.log(1 + T.exp(rho))

    def std_to_log(self, sigma):
        """Reverse log_to_std transformation."""
        return np.log(np.exp(sigma) - 1)

    def get_W_shape(self):
        raise NotImplementedError(
            "Method should be implemented in subclass.")

    def get_b_shape(self):
        raise NotImplementedError(
            "Method should be implemented in subclass.")

    def save_old_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old.set_value(self.mu.get_value())
        self.rho_old.set_value(self.rho.get_value())
        self.b_mu_old.set_value(self.b_mu.get_value())
        self.b_rho_old.set_value(self.b_rho.get_value())

    def reset_to_old_params(self):
        """Reset to old parameter values for KL calculation."""
        self.mu.set_value(self.mu_old.get_value())
        self.rho.set_value(self.rho_old.get_value())
        self.b_mu.set_value(self.b_mu_old.get_value())
        self.b_rho.set_value(self.b_rho_old.get_value())

    def get_W(self):
        mask = 0 if self.disable_variance else 1
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=self.get_W_shape(), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        return self.mu + self.log_to_std(self.rho) * epsilon * mask

    def get_b(self):
        mask = 0 if self.disable_variance else 1
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=self.get_b_shape(), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        return self.b_mu + self.log_to_std(self.b_rho) * epsilon * mask

    def kl_div_new_old(self):
        return self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), self.mu_old, self.log_to_std(self.rho_old))

    def kl_div_old_new(self):
        return self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), self.mu, self.log_to_std(self.rho))

    def kl_div_new_prior(self):
        return self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), 0., self.prior_sd)

    def kl_div_old_prior(self):
        return self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), 0., self.prior_sd)

    def kl_div_prior_new(self):
        return self.kl_div_p_q(
            0., self.prior_sd, self.mu,  self.log_to_std(self.rho))

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        numerator = T.square(p_mean - q_mean) + \
            T.square(p_std) - T.square(q_std)
        denominator = 2 * T.square(q_std) + 1e-8
        return T.sum(
            numerator / denominator + T.log(q_std) - T.log(p_std))


class BayesianConvLayer(BayesianLayer):
    """Bayesian Convolutional layer"""

    def __init__(
        self,
        incoming,
        num_filters,
        filter_size,
        stride=(1, 1),
        pad=0,
        untie_biases=False,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.rectify,
        flip_filters=True,
        prior_sd=None,
        **kwargs
    ):
        super(BayesianConvLayer, self).__init__(
            incoming, num_filters, nonlinearity, prior_sd, **kwargs)

        self.n = len(self.input_shape) - 2
        self.nonlinearity = nonlinearity
        self.filter_size = lasagne.utils.as_tuple(filter_size, self.n, int)
        self.flip_filters = flip_filters
        self.stride = lasagne.utils.as_tuple(stride, self.n, int)
        self.untie_biases = untie_biases

        self.init_params()

        if pad == 'same':
            if any(s % 2 == 0 for s in self.filter_size):
                raise NotImplementedError(
                    '`same` padding requires odd filter size.')
        if pad == 'valid':
            self.pad = lasagne.utils.as_tuple(0, self.n)
        elif pad in ('full', 'same'):
            self.pad = pad
        else:
            self.pad = lasagne.utils.as_tuple(pad, self.n, int)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (self.num_units, num_input_channels) + self.filter_size

    def get_b_shape(self):
        if self.untie_biases:
            return (self.num_units,) + self.output_shape[2:]
        else:
            return (self.num_units,)

    def get_output_shape_for(self, input_shape):
        pad = self.pad if isinstance(self.pad, tuple) else (self.pad,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_units) +
                tuple(lasagne.layers.conv.conv_output_length(input, filter, stride, p)
                      for input, filter, stride, p
                      in zip(input_shape[2:], self.filter_size,
                             self.stride, pad)))

    def convolve(self, input, **kwargs):
        # Input should be (batch_size, n_in_filters, img_h, img_w).
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = T.nnet.conv2d(input, self.get_W(),
                               self.input_shape, self.get_W_shape(),
                               subsample=self.stride,
                               border_mode=border_mode,
                               filter_flip=self.flip_filters)
        return conved

    def get_output_for(self, input, **kwargs):
        conved = self.convolve(input, **kwargs)

        if self.untie_biases:
            activation = conved + T.shape_padleft(self.get_b(), 1)
        else:
            activation = conved + \
                self.get_b().dimshuffle(('x', 0) + ('x',) * self.n)

        return self.nonlinearity(activation)


class BayesianDenseLayer(BayesianLayer):
    """Probabilistic layer that uses Gaussian weights.

    Each weight has two parameters: mean and standard deviation (std).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 prior_sd=None,
                 use_local_reparametrization_trick=None,
                 group_variance_by=None,
                 disable_variance=None,
                 **kwargs):
        super(BayesianDenseLayer, self).__init__(
            incoming, num_units, nonlinearity, prior_sd, **kwargs)

        self.group_variance_by = group_variance_by
        self.use_local_reparametrization_trick = use_local_reparametrization_trick

        self.init_params()

    def get_W_shape(self):
        return (self.num_inputs, self.num_units)

    def get_b_shape(self):
        return (self.num_units,)

    def get_output_for_reparametrization(self, input, **kwargs):
        """Implementation of the local reparametrization trick.

        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.

        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """
        mask = 0 if self.disable_variance else 1
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        gamma = T.dot(input, self.mu) + self.b_mu.dimshuffle('x', 0)
        delta = T.dot(T.square(input), T.square(self.log_to_std(
            self.rho))) + T.square(self.log_to_std(self.b_rho)).dimshuffle('x', 0)

        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable

        activation = gamma + T.sqrt(delta) * epsilon * mask

        return self.nonlinearity(activation)

    def get_output_for(self, input, **kwargs):
        if self.use_local_reparametrization_trick:
            return self.get_output_for_reparametrization(input, **kwargs)
        else:
            return self.get_output_for_default(input, **kwargs)

    def get_output_for_default(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.get_W()) + \
            self.get_b().dimshuffle('x', 0)

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class ConvBNN(LasagnePowered, Serializable):
    """Convolutional Bayesian neural network (ConvBNN), according to Blundell2016."""

    # Enums
    GroupVarianceBy = enum(WEIGHT='weight', UNIT='unit', LAYER='layer')
    OutputType = enum(REGRESSION='regression', CLASSIFICATION='classfication')
    SurpriseType = enum(
        INFGAIN='information gain', COMPR='compression gain', BALD='BALD')

    def __init__(self,
                 layers_disc,
                 n_out,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 likelihood_sd=5.0,
                 second_order_update=False,
                 learning_rate=0.0001,
                 compression=False,
                 information_gain=True,
                 update_prior=False,
                 update_likelihood_sd=False
                 ):

        Serializable.quick_init(self, locals())

        self.batch_size = batch_size
        self.transf = trans_func
        self.n_out = n_out
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_disc = layers_disc
        self.n_batches = n_batches
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.likelihood_sd = likelihood_sd
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.compression = compression
        self.information_gain = information_gain
        self.update_prior = update_prior
        self.update_likelihood_sd = update_likelihood_sd

        assert self.information_gain or self.compression

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()

    def save_old_params(self):
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.save_old_params()
        self.old_likelihood_sd.set_value(self.likelihood_sd.get_value())

    def reset_to_old_params(self):
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.reset_to_old_params()
        self.likelihood_sd.set_value(self.old_likelihood_sd.get_value())

    def compression_improvement(self):
        """KL divergence KL[old_param||new_param]"""
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def surprise(self):
        surpr = 0.
        if self.compression:
            surpr += self.compression_improvement()
        if self.information_gain:
            surpr += self.inf_gain()
        return surpr

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianDenseLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_prior_new() for l in layers)

    def _log_prob_normal(self, input, mu=0., sigma=1.):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def loss(self, input, target, likelihood_sd):

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, likelihood_sd))
        log_p_D_given_w = sum(_log_p_D_given_w)
        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        if self.update_prior:
            kl = self.kl_div()
        else:
            kl = self.log_p_w_q_w_kl()
        if self.use_reverse_kl_reg:
            kl += self.reverse_kl_reg_factor * \
                self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
        return kl / self.n_batches - log_p_D_given_w / self.n_samples

    def loss_last_sample(self, input, target, likelihood_sd):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """
        # Fix sampled noise.
        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(sample|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, likelihood_sd))
        log_p_D_given_w = sum(_log_p_D_given_w)
        # Calculate loss function.
        # self.kl_div() should be zero when taking second order step
        return self.kl_div() - log_p_D_given_w / self.n_samples

    def dbg_nll(self, input, target, likelihood_sd):
        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(sample|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, likelihood_sd))
        log_p_D_given_w = sum(_log_p_D_given_w)
        return - log_p_D_given_w / self.n_samples

    def build_network(self):

        # Layers
        for i, layer_disc in enumerate(self.layers_disc):
            if i == 0:
                assert(layer_disc['name'] == 'input')

            print(layer_disc)
            if layer_disc['name'] == 'input':
                network = lasagne.layers.InputLayer(
                    shape=layer_disc['in_shape'])
            elif layer_disc['name'] == 'convolution':
                network = BayesianConvLayer(network, num_filters=layer_disc[
                    'n_filters'], filter_size=layer_disc['filter_size'], prior_sd=self.prior_sd)
            elif layer_disc['name'] == 'pool':
                network = PoolLayer(network, pool_size=layer_disc['pool_size'])
            elif layer_disc['name'] == 'gaussian':
                network = BayesianDenseLayer(
                    network, num_units=layer_disc[
                        'n_units'], nonlinearity=self.transf, prior_sd=self.prior_sd,
                    use_local_reparametrization_trick=True)
            elif layer_disc['name'] == 'deterministic':
                network = lasagne.layers.DenseLayer(
                    network, num_units=layer_disc['n_units'], nonlinearity=self.transf)
            elif layer_disc['name'] == 'transconvolution':
                network = TransConvLayer(network, num_filters=layer_disc[
                    'n_filters'], filter_size=layer_disc['filter_size'])
            elif layer_disc['name'] == 'upscale':
                network = UpscaleLayer(
                    network, scale_factor=layer_disc['scale_factor'])
            else:
                raise(Exception('Unknown layer!'))

        # Output layer
        network = BayesianDenseLayer(
            network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, use_local_reparametrization_trick=True)

        self.network = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.tensor4('inputs',
                              dtype=theano.config.floatX)  # @UndefinedVariable
        target_var = T.matrix('targets',
                              dtype=theano.config.floatX)  # @UndefinedVariable

        # Make the likelihood standard deviation a trainable parameter.
        self.likelihood_sd = theano.shared(
            value=1.0,  # self.likelihood_sd_init,
            name='likelihood_sd'
        )
        self.old_likelihood_sd = theano.shared(
            value=1.0,  # self.likelihood_sd_init,
            name='old_likelihood_sd'
        )

        # Loss function.
        loss = self.loss(input_var, target_var, self.likelihood_sd)
        loss_only_last_sample = self.loss_last_sample(
            input_var, target_var, self.likelihood_sd)

        # Create update methods.
        params_kl = lasagne.layers.get_all_params(self.network, trainable=True)
        params = []
        params.extend(params_kl)
        if self.update_likelihood_sd:
            params.append(self.likelihood_sd)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=self.learning_rate)

        # Train/val fn.
        self.pred_fn = ext.compile_function(
            [input_var], self.pred_sym(input_var), log_name='fn_pred')
        # We want to resample when actually updating the BNN itself, otherwise
        # you will fit to the specific noise.
        self.train_fn = ext.compile_function(
            [input_var, target_var], loss, updates=updates, log_name='fn_train')

        if self.second_order_update:

            oldparams = lasagne.layers.get_all_params(
                self.network, oldparam=True)
            step_size = T.scalar('step_size',
                                 dtype=theano.config.floatX)  # @UndefinedVariable

            def second_order_update(loss, params, oldparams, step_size):
                """Second-order update method for optimizing loss_last_sample, so basically,
                KL term (new params || old params) + NLL of latest sample. The Hessian is
                evaluated at the origin and provides curvature information to make a more
                informed step in the correct descent direction."""
                grads = theano.grad(loss, params)
                updates = OrderedDict()

                for i in xrange(len(params)):
                    param = params[i]
                    grad = grads[i]

                    if param.name == 'mu' or param.name == 'b_mu':
                        oldparam_rho = oldparams[i + 1]
                        invH = T.square(T.log(1 + T.exp(oldparam_rho)))
                    elif param.name == 'rho' or param.name == 'b_rho':
                        oldparam_rho = oldparams[i]
                        p = param
                        H = 2. * (T.exp(2 * p)) / \
                            (1 + T.exp(p))**2 / (T.log(1 + T.exp(p))**2)
                        invH = 1. / H
                    elif param.name == 'likelihood_sd':
                        invH = 0.
                    # So wtf is going wrong here?
                    updates[param] = param - step_size * invH * grad

                return updates

            # DEBUG
            # -----
            def debug_H(loss, params, oldparams):
                grads = theano.grad(loss, params)
                updates = OrderedDict()

                invHs = []
                for i in xrange(len(params)):
                    param = params[i]
                    grad = grads[i]

                    if param.name == 'mu' or param.name == 'b_mu':
                        oldparam_rho = oldparams[i + 1]
                        invH = T.square(T.log(1 + T.exp(oldparam_rho)))
                    elif param.name == 'rho' or param.name == 'b_rho':
                        oldparam_rho = oldparams[i]
                        p = param
                        H = 2. * (T.exp(2 * p)) / \
                            (1 + T.exp(p))**2 / (T.log(1 + T.exp(p))**2)
                        invH = 1. / H
#                     elif param.name == 'likelihood_sd':
#                         invH = 0.
                    invHs.append(invH)
                return invHs

            def debug_g(loss, params, oldparams):
                grads = theano.grad(loss, params)
                updates = OrderedDict()

                invHs = []
                for i in xrange(len(params)):
                    param = params[i]
                    grad = grads[i]

                    if param.name == 'mu' or param.name == 'b_mu':
                        oldparam_rho = oldparams[i + 1]
                        invH = T.square(T.log(1 + T.exp(oldparam_rho)))
                    elif param.name == 'rho' or param.name == 'b_rho':
                        oldparam_rho = oldparams[i]
                        p = param
                        H = 2. * (T.exp(2 * p)) / \
                            (1 + T.exp(p))**2 / (T.log(1 + T.exp(p))**2)
                        invH = 1. / H
#                     elif param.name == 'likelihood_sd':
#                         invH = 0.
                    invHs.append(invH)
                return grads
            # -----

            def fast_kl_div(loss, params, oldparams, step_size):

                grads = T.grad(loss, params)

                kl_component = []
                for i in xrange(len(params)):
                    param = params[i]
                    grad = grads[i]

                    if param.name == 'mu' or param.name == 'b_mu':
                        oldparam_rho = oldparams[i + 1]
                        invH = T.square(T.log(1 + T.exp(oldparam_rho)))
                    elif param.name == 'rho' or param.name == 'b_rho':
                        oldparam_rho = oldparams[i]
                        p = param
                        H = 2. * (T.exp(2 * p)) / \
                            (1 + T.exp(p))**2 / (T.log(1 + T.exp(p))**2)
                        invH = 1. / H
                    elif param.name == 'likelihood_sd':
                        invH = 0.

                    kl_component.append(
                        T.sum(0.5 * T.square(step_size) * T.square(grad) * invH))

                return sum(kl_component)

            compute_fast_kl_div = fast_kl_div(
                loss_only_last_sample, params, oldparams, step_size)

            self.train_update_fn = ext.compile_function(
                [input_var, target_var, step_size], compute_fast_kl_div, log_name='fn_surprise_fast', no_default_updates=False)

            # Code to actually perform second order updates
            # ---------------------------------------------
#             updates_kl = second_order_update(
#                 loss_only_last_sample, params, oldparams, step_size)
#
#             self.train_update_fn = ext.compile_function(
#                 [input_var, target_var, step_size], loss_only_last_sample, updates=updates_kl, log_name='fn_surprise_2nd', no_default_updates=False)
            # ---------------------------------------------

            # DEBUG
            # -----
#             self.debug_H = ext.compile_function(
#                 [input_var, target_var], debug_H(
#                     loss_only_last_sample, params, oldparams),
#                 log_name='fn_debug_grads')
#             self.debug_g = ext.compile_function(
#                 [input_var, target_var], debug_g(
#                     loss_only_last_sample, params, oldparams),
#                 log_name='fn_debug_grads')
            # -----

        else:
            # Use SGD to update the model for a single sample, in order to
            # calculate the surprise.

            def sgd(loss, params, learning_rate):
                grads = theano.grad(loss, params)
                updates = OrderedDict()
                for param, grad in zip(params, grads):
                    if param.name == 'likelihood_sd':
                        updates[param] = param  # - learning_rate * grad
                    else:
                        updates[param] = param - learning_rate * grad

                return updates

            updates_kl = sgd(
                loss_only_last_sample, params, learning_rate=self.learning_rate)

            self.train_update_fn = ext.compile_function(
                [input_var, target_var], loss_only_last_sample, updates=updates_kl, log_name='fn_surprise_1st', no_default_updates=False)

        self.eval_loss = ext.compile_function(
            [input_var, target_var], loss,  log_name='fn_eval_loss', no_default_updates=False)

        # DEBUG
        # -----
#         # Calculate surprise.
#         self.fn_surprise = ext.compile_function(
#             [], self.surprise(), log_name='fn_surprise')
        self.fn_dbg_nll = ext.compile_function(
            [input_var, target_var], self.dbg_nll(input_var, target_var, self.likelihood_sd), log_name='fn_dbg_nll', no_default_updates=False)
        self.fn_kl = ext.compile_function(
            [], self.kl_div(), log_name='fn_kl')
        self.fn_kl_from_prior = ext.compile_function(
            [], self.log_p_w_q_w_kl(), log_name='fn_kl_from_prior')
        # -----

if __name__ == '__main__':
    pass
