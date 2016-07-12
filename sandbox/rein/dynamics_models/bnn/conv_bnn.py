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

    def num_weights(self):
        return np.prod(self.get_W_shape())

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


class BayesianDeConvLayer(BayesianLayer):

    def __init__(self,
                 incoming,
                 num_filters,
                 filter_size,
                 stride=(1, 1),
                 crop=0,
                 untie_biases=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 flip_filters=False,
                 prior_sd=None,
                 **kwargs):

        super(BayesianDeConvLayer, self).__init__(
            incoming, num_filters, nonlinearity, prior_sd, **kwargs)

        pad = crop
        self.crop = crop
        self.n = len(self.input_shape) - 2
        self.nonlinearity = nonlinearity
        self.num_units = num_filters
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
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_units) + self.filter_size

    def get_b_shape(self):
        if self.untie_biases:
            return (self.num_units,) + self.output_shape[2:]
        else:
            return (self.num_units,)

    def get_output_shape_for(self, input_shape):
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_units) +
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
        conved = op(self.get_W(), input, output_size)
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
    """(Convolutional) Bayesian neural network (BNN), according to Blundell2016.

    The input and output to the network is a flat array. Internally, the shapes of input_dim 
    and output_dim are used. Use layers_disc to describe the layers between the input and output
    layers.
    """

    # Enums
    GroupVarianceBy = enum(WEIGHT='weight', UNIT='unit', LAYER='layer')
    OutputType = enum(REGRESSION='regression', CLASSIFICATION='classfication')
    SurpriseType = enum(
        INFGAIN='information gain', COMPR='compression gain', BALD='BALD')

    def __init__(self,
                 input_dim,
                 output_dim,
                 layers_disc,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 second_order_update=False,
                 learning_rate=0.0001,
                 surprise_type=SurpriseType.INFGAIN,
                 update_prior=False,
                 update_likelihood_sd=False,
                 group_variance_by=GroupVarianceBy.WEIGHT,
                 use_local_reparametrization_trick=True,
                 likelihood_sd_init=1.0,
                 output_type=OutputType.REGRESSION,
                 num_classes=None,
                 num_output_dim=None,
                 disable_variance=False,
                 debug=False
                 ):

        Serializable.quick_init(self, locals())

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_disc = layers_disc
        self.n_batches = n_batches
        self.likelihood_sd_init = likelihood_sd_init
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.surprise_type = surprise_type
        self.update_prior = update_prior
        self.update_likelihood_sd = update_likelihood_sd
        self.group_variance_by = group_variance_by
        self.use_local_reparametrization_trick = use_local_reparametrization_trick
        self.output_type = output_type
        self.num_classes = num_classes
        self.num_output_dim = num_output_dim
        self.disable_variance = disable_variance
        self.debug = debug

        if self.output_type == ConvBNN.OutputType.CLASSIFICATION:
            assert self.num_classes is not None
            assert self.num_output_dim is not None
            assert self.n_out == self.num_classes * self.num_output_dim

        if self.group_variance_by == ConvBNN.GroupVarianceBy.LAYER and self.use_local_reparametrization_trick:
            print(
                'Setting use_local_reparametrization_trick=True cannot be used with group_variance_by==\'layer\', changing to False')
            self.use_local_reparametrization_trick = False

        if self.output_type == ConvBNN.OutputType.CLASSIFICATION and self.update_likelihood_sd:
            print(
                'Setting output_type=\'classification\' cannot be used with update_likelihood_sd=True, changing to False.')
            self.update_likelihood_sd = False

        if self.disable_variance:
            print('Warning: all noise has been disabled, only using means.')

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()

        print('num_weights: {}'.format(self.num_weights()))

    def save_old_params(self):
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.save_old_params()
        if self.update_likelihood_sd:
            self.old_likelihood_sd.set_value(self.likelihood_sd.get_value())

    def reset_to_old_params(self):
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.reset_to_old_params()
        if self.update_likelihood_sd:
            self.likelihood_sd.set_value(self.old_likelihood_sd.get_value())

    def compression_improvement(self):
        """KL divergence KL[old_param||new_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def num_weights(self):
        print('Disclaimer: only work with BNNLayers!')
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.num_weights() for l in layers)

    def ent(self, input):
        # FIXME: work in progress
        mtrx_pred = np.zeros((self.n_samples, self.n_out))
        for i in xrange(self.n_samples):
            # Make prediction.
            mtrx_pred[i] = self.pred_fn(input)
        cov = np.cov(mtrx_pred, rowvar=0)
        if isinstance(cov, float):
            var = np.trace(cov) / float(cov.shape[0])
        else:
            var = cov
        return var

    def entropy(self, input, likelihood_sd, **kwargs):
        """ Entropy of a batch of input/output samples. """

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            for _ in xrange(self.n_samples):
                sampled_mean = self.pred_sym(input)
                # Calculate model likelihood log(P(D|w)).
                if self.output_type == ConvBNN.OutputType.CLASSIFICATION:
                    lh = self.likelihood_classification(
                        sampled_mean, prediction)
                elif self.output_type == ConvBNN.OutputType.REGRESSION:
                    lh = self.likelihood_regression(
                        sampled_mean, prediction, likelihood_sd)
                _log_p_D_given_w.append(lh)
        log_p_D_given_w = sum(_log_p_D_given_w)

        return - log_p_D_given_w / (self.n_samples)**2 + 0.5 * (np.log(2 * np.pi
                                                                       * likelihood_sd**2) + 1)

    def surprise(self, **kwargs):

        if self.surprise_type == ConvBNN.SurpriseType.COMPR:
            surpr = self.compression_improvement()
        elif self.surprise_type == ConvBNN.SurpriseType.INFGAIN:
            surpr = self.inf_gain()
        elif self.surprise_type == ConvBNN.SurpriseType.BALD:
            surpr = self.entropy(**kwargs)
        else:
            raise Exception(
                'Uknown surprise_type {}'.format(self.surprise_type))
        return surpr

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_prior_new() for l in layers)

    def _log_prob_normal(self, input, mu=0., sigma=1.):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def likelihood_regression(self, target, prediction, likelihood_sd):
        return self._log_prob_normal(
            target, prediction, likelihood_sd)

    def likelihood_classification(self, target, prediction):
        # Cross-entropy; target vector selecting correct prediction
        # entries.

        # Numerical stability.
        prediction += 1e-8

        target2 = target + T.arange(target.shape[1]) * self.num_classes
        target3 = target2.T.ravel()
        idx = T.arange(target.shape[0])
        idx2 = T.tile(idx, self.num_output_dim)
        prediction_selected = prediction[
            idx2, target3].reshape([self.num_output_dim, target.shape[0]]).T
        ll = T.sum(T.log(prediction_selected))
        return ll

    def loss(self, input, target, kl_factor=1.0, disable_kl=False, **kwargs):

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.output_type == ConvBNN.OutputType.CLASSIFICATION:
                lh = self.likelihood_classification(target, prediction)
            elif self.output_type == ConvBNN.OutputType.REGRESSION:
                lh = self.likelihood_regression(target, prediction, **kwargs)
            else:
                raise Exception(
                    'Uknown output_type {}'.format(self.output_type))
            _log_p_D_given_w.append(lh)
        log_p_D_given_w = sum(_log_p_D_given_w)

        if disable_kl:
            return - log_p_D_given_w / self.n_samples
        else:
            if self.update_prior:
                kl = self.kl_div()
            else:
                kl = self.log_p_w_q_w_kl()
            return kl / self.n_batches * kl_factor - log_p_D_given_w / self.n_samples

    def loss_last_sample(self, input, target, **kwargs):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """
        return self.loss(input, target, disable_kl=True, **kwargs)

    def build_network(self):

        # Input to the network is always flattened.
        network = lasagne.layers.InputLayer(
            shape=(None, np.prod(self.input_dim)))
        # Reshape according to the input_dim
        network = lasagne.layers.reshape(network, ([0],) + self.input_dim)

        for i, layer_disc in enumerate(self.layers_disc):

            if layer_disc['name'] == 'convolution':
                network = BayesianConvLayer(network, num_filters=layer_disc[
                    'n_filters'], filter_size=layer_disc['filter_size'], prior_sd=self.prior_sd, stride=layer_disc['stride'])
            elif layer_disc['name'] == 'pool':
                network = lasagne.layers.Pool2DLayer(
                    network, pool_size=layer_disc['pool_size'])
            elif layer_disc['name'] == 'pad':
                network = lasagne.layers.PadLayer(
                    network, val=0)
            elif layer_disc['name'] == 'gaussian':
                network = BayesianDenseLayer(
                    network, num_units=layer_disc[
                        'n_units'], nonlinearity=self.transf, prior_sd=self.prior_sd,
                    use_local_reparametrization_trick=True)
            elif layer_disc['name'] == 'deterministic':
                network = lasagne.layers.DenseLayer(
                    network, num_units=layer_disc['n_units'], nonlinearity=self.transf)
            elif layer_disc['name'] == 'deconvolution':
                network = BayesianDeConvLayer(network, num_filters=layer_disc[
                    'n_filters'], filter_size=layer_disc['filter_size'], prior_sd=self.prior_sd, stride=layer_disc['stride'])
            elif layer_disc['name'] == 'upscale':
                network = lasagne.layers.Upscale2DLayer(
                    network, scale_factor=layer_disc['scale_factor'])
            elif layer_disc['name'] == 'reshape':
                network = lasagne.layers.ReshapeLayer(
                    network, shape=layer_disc['shape'])

            else:
                raise(Exception('Unknown layer!'))

            print('layer {}: {}\n\toutsize: {}'.format(
                i, layer_disc, network.output_shape))

        # Output of output_dim is flattened again.
        self.network = lasagne.layers.flatten(network)

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        kl_factor = T.scalar('kl_factor',
                             dtype=theano.config.floatX)
        # Assume all inputs are flattened.
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)

        if self.output_type == ConvBNN.OutputType.REGRESSION:
            target_var = T.matrix('targets',
                                  dtype=theano.config.floatX)

            # Make the likelihood standard deviation a trainable parameter.
            self.likelihood_sd = theano.shared(
                value=self.likelihood_sd_init,  # self.likelihood_sd_init,
                name='likelihood_sd'
            )
            self.old_likelihood_sd = theano.shared(
                value=self.likelihood_sd_init,  # self.likelihood_sd_init,
                name='old_likelihood_sd'
            )

            # Loss function.
            loss = self.loss(
                input_var, target_var, kl_factor, likelihood_sd=self.likelihood_sd)
            loss_only_last_sample = self.loss_last_sample(
                input_var, target_var, likelihood_sd=self.likelihood_sd)

        elif self.output_type == ConvBNN.OutputType.CLASSIFICATION:

            target_var = T.imatrix('targets')

            # Loss function.
            loss = self.loss(
                input_var, target_var, kl_factor)
            loss_only_last_sample = self.loss_last_sample(
                input_var, target_var)

        else:
            raise Exception(
                'Unknown self.output_type {}'.format(self.output_type))

        # Create update methods.
        params_kl = lasagne.layers.get_all_params(self.network, trainable=True)
        params = []
        params.extend(params_kl)
        if self.output_type == 'regression' and self.update_likelihood_sd:
            # No likelihood sd for classification tasks.
            params.append(self.likelihood_sd)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=self.learning_rate)

        # Train/val fn.
        self.pred_fn = ext.compile_function(
            [input_var], self.pred_sym(input_var), log_name='fn_pred')
        # We want to resample when actually updating the BNN itself, otherwise
        # you will fit to the specific noise.
        self.train_fn = ext.compile_function(
            [input_var, target_var, kl_factor], loss, updates=updates, log_name='fn_train')

        if self.surprise_type == ConvBNN.SurpriseType.INFGAIN:
            if self.second_order_update:

                oldparams = lasagne.layers.get_all_params(
                    self.network, oldparam=True)
                step_size = T.scalar('step_size',
                                     dtype=theano.config.floatX)

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
                        updates[param] = param - step_size * invH * grad

                    return updates

                def fast_kl_div(loss, params, oldparams, step_size):
                    # FIXME: doesn't work yet for group_variance_by!='weight'.

                    grads = T.grad(loss, params)

                    kl_component = []
                    for i in xrange(len(params)):
                        param = params[i]
                        grad = grads[i]

                        if param.name == 'mu' or param.name == 'b_mu':
                            oldparam_rho = oldparams[i + 1]
                            if self.group_variance_by == 'unit':
                                if not isinstance(oldparam_rho, float):
                                    oldparam_rho = oldparam_rho.dimshuffle(
                                        0, 'x')
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

        elif self.surprise_type == ConvBNN.SurpriseType.BALD:
            # BALD
            self.train_update_fn = ext.compile_function(
                [input_var], self.surprise(input=input_var, likelihood_sd=self.likelihood_sd), log_name='fn_surprise_bald')

if __name__ == '__main__':
    pass
