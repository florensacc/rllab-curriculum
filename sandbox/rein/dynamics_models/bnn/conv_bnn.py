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
                 matrix_variate_gaussian=False,
                 mvg_rank=1,
                 **kwargs):
        super(BayesianLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        if nonlinearity == 'rbf':
            self.c = self.add_param(lasagne.init.GlorotUniform(), self.num_units, name='c')
            self.s = self.add_param(lasagne.init.GlorotUniform(), self.num_units, name='s')

            def rbf(x):
                C = self.c[np.newaxis, :, :]
                X = x[:, np.newaxis, :]
                difnorm = T.sum((C - X) ** 2, axis=-1)
                return T.exp(-difnorm * (self.s ** 2))

            self.nonlinearity = rbf
        else:
            self.nonlinearity = nonlinearity

        self.prior_sd = prior_sd
        self.num_units = num_units
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.prior_rho = self.inv_softplus(self.prior_sd)
        print('prior_rho: {}'.format(self.prior_rho))
        self.disable_variance = disable_variance
        self._matrix_variate_gaussian = matrix_variate_gaussian
        self.mvg_rank = mvg_rank

        self.mu_tmp, self.b_mu_tmp, self.rho_tmp, self.b_rho_tmp = None, None, None, None
        self.mu, self.rho, self.b_mu, self.b_rho = None, None, None, None
        self.mu_old, self.rho_old, self.b_mu_old, self.b_rho_old = None, None, None, None

        if self.disable_variance:
            print('Variance disabled!')

    def init_params(self):

        if self._matrix_variate_gaussian:
            # Weight distributions are parametrized according to Louizos2016,
            # which is a combination of the previous out and in variance
            # tying.
            self.mu = self.add_param(
                lasagne.init.Normal(0.1, 0.), self.get_W_shape(), name='mu', bayesian=True)
            # So we should have weights (self.num_units + self.num_inputs,)
            # instead of (self.num_units, self.num_inputs).
            if not self.disable_variance:
                self.rho = self.add_param(
                    lasagne.init.Constant(self.inv_softplus(np.sqrt(self.prior_sd))),
                    ((self.num_inputs + self.num_units) * self.mvg_rank,),
                    name='rho', bayesian=True)

            self.b_mu = self.add_param(
                lasagne.init.Constant(0), self.get_b_shape(), name="b_mu", regularizable=False, bayesian=True)
            # Single bias variance, since outgoing weights are tied.
            if not self.disable_variance:
                self.b_rho = self.add_param(
                    lasagne.init.Constant(self.prior_rho),
                    (1,),
                    name="b_rho",
                    regularizable=False, bayesian=True
                )

            # Backup params for KL calculations.
            self.mu_old = self.add_param(
                np.zeros(self.get_W_shape()), self.get_W_shape(), name='mu_old', trainable=False, oldparam=True)
            if not self.disable_variance:
                self.rho_old = self.add_param(
                    lasagne.init.Constant(self.prior_rho),
                    ((self.num_inputs + self.num_units) * self.mvg_rank,),
                    name='rho_old', trainable=False, oldparam=True)

            # Bias priors.
            self.b_mu_old = self.add_param(
                np.zeros(self.get_b_shape()), self.get_b_shape(), name="b_mu_old", regularizable=False,
                trainable=False, oldparam=True)
            if not self.disable_variance:
                self.b_rho_old = self.add_param(
                    lasagne.init.Constant(self.prior_rho),
                    (1,), name="b_rho_old", regularizable=False,
                    trainable=False, oldparam=True)

        else:
            # In fact, this should be initialized to np.zeros(self.get_W_shape()),
            # but this trains much slower.
            self.mu = self.add_param(
                # lasagne.init.Normal(0.00001, 0),
                lasagne.init.GlorotUniform(),
                self.get_W_shape(), name='mu', bayesian=True)
            if not self.disable_variance:
                self.rho = self.add_param(
                    lasagne.init.Constant(self.prior_rho), self.get_W_shape(), name='rho', bayesian=True)

            # TODO: Perhaps biases should have a postive value, to avoid zeroing the
            # relus.
            self.b_mu = self.add_param(
                lasagne.init.Constant(0), self.get_b_shape(), name="b_mu", regularizable=False, bayesian=True)
            if not self.disable_variance:
                self.b_rho = self.add_param(
                    lasagne.init.Constant(self.prior_rho), self.get_b_shape(), name="b_rho", regularizable=False,
                    bayesian=True)

            # Backup params for KL calculations.
            self.mu_old = self.add_param(
                np.zeros(self.get_W_shape()), self.get_W_shape(), name='mu_old', trainable=False, oldparam=True)
            if not self.disable_variance:
                self.rho_old = self.add_param(
                    np.zeros(self.get_W_shape()), self.get_W_shape(), name='rho_old', trainable=False, oldparam=True)

            # Bias priors.
            self.b_mu_old = self.add_param(
                np.zeros(self.get_b_shape()), self.get_b_shape(), name="b_mu_old", regularizable=False,
                trainable=False, oldparam=True)
            if not self.disable_variance:
                self.b_rho_old = self.add_param(
                    np.zeros(self.get_b_shape()), self.get_b_shape(), name="b_rho_old", regularizable=False,
                    trainable=False, oldparam=True)

    def num_weights(self):
        return np.prod(self.get_W_shape())

    def softplus(self, rho):
        """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+

        This makes sure that we don't get negative stds. However, a downside might be
        that we have little gradient on close to 0 std (= -inf using this transformation).
        """
        return T.log(1 + T.exp(rho))

    def inv_softplus(self, sigma):
        """Reverse softplus transformation."""
        return np.log(np.exp(sigma) - 1)

    def get_W_shape(self):
        raise NotImplementedError(
            "Method should be implemented in subclass.")

    def get_b_shape(self):
        raise NotImplementedError(
            "Method should be implemented in subclass.")

    def save_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old.set_value(self.mu.get_value())
        self.b_mu_old.set_value(self.b_mu.get_value())
        if not self.disable_variance:
            self.rho_old.set_value(self.rho.get_value())
            self.b_rho_old.set_value(self.b_rho.get_value())

    def load_prev_params(self):
        """Reset to old parameter values for KL calculation."""
        self.mu_tmp = self.mu.get_value()
        self.b_mu_tmp = self.b_mu.get_value()
        if not self.disable_variance:
            self.rho_tmp = self.rho.get_value()
            self.b_rho_tmp = self.b_rho.get_value()

        self.mu.set_value(lasagne.utils.floatX(self.mu_old.get_value()))
        self.b_mu.set_value(lasagne.utils.floatX(self.b_mu_old.get_value()))
        if not self.disable_variance:
            self.rho.set_value(lasagne.utils.floatX(self.rho_old.get_value()))
            self.b_rho.set_value(lasagne.utils.floatX(self.b_rho_old.get_value()))

    def load_cur_params(self):
        """Reset to old parameter values for KL calculation."""
        self.mu_tmp = self.mu_old.get_value()
        self.b_mu_tmp = self.b_mu_old.get_value()
        if not self.disable_variance:
            self.rho_tmp = self.rho_old.get_value()
            self.b_rho_tmp = self.b_rho_old.get_value()

        self.mu.set_value(self.mu_tmp)
        self.b_mu.set_value(self.b_mu_tmp)
        if self.disable_variance:
            self.rho.set_value(self.rho_tmp)
            self.b_rho.set_value(self.b_rho_tmp)

    def get_W_full(self):
        # W = M + U ^ 0.5 * E * V ^ 0.5
        pass

    def get_W(self):
        if self._matrix_variate_gaussian:
            if not self.disable_variance:
                s = self.softplus(self.rho)
                s_u = s[self.num_inputs:]
                s_v = s[:self.num_inputs]
                s_v = s[:self.num_inputs]
                s = T.dot(s_u.dimshuffle(0, 'x'), s_v.dimshuffle('x', 0)).T
                epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=1.,
                                            dtype=theano.config.floatX)
                return self.mu + s * epsilon
            else:
                return self.mu
        else:
            if not self.disable_variance:
                epsilon = self._srng.normal(size=self.get_W_shape(), avg=0., std=1., dtype=theano.config.floatX)
                return self.mu + self.softplus(self.rho) * epsilon
            else:
                return self.mu

    def get_b(self):
        if self._matrix_variate_gaussian:
            if not self.disable_variance:
                epsilon = self._srng.normal(size=(1,), avg=0., std=1., dtype=theano.config.floatX)
                return self.b_mu + T.mean(self.softplus(self.b_rho)) * epsilon
            else:
                return self.b_mu
        else:
            if not self.disable_variance:
                epsilon = self._srng.normal(size=self.get_b_shape(), avg=0., std=1., dtype=theano.config.floatX)
                return self.b_mu + self.softplus(self.b_rho) * epsilon
            else:
                return self.b_mu

    def l1_new_old(self):
        l1_a = T.abs_((self.mu - self.mu_old).flatten())
        l1_b = T.abs_((self.b_mu - self.b_mu_old).flatten())
        if not self.disable_variance:
            l1_c = T.abs_((self.rho - self.rho_old).flatten())
            l1_d = T.abs_((self.b_rho - self.b_rho_old).flatten())
            return T.concatenate((l1_a, l1_b, l1_c, l1_d))
        else:
            return T.concatenate((l1_a, l1_b))

    # We don't calculate the KL for biases, as they should be able to
    # arbitrarily shift.
    def kl_div_new_old(self):
        return self.kl_div_p_q(self.mu, self.softplus(self.rho), self.mu_old, self.softplus(self.rho_old))

    def kl_div_old_new(self):
        return self.kl_div_p_q(self.mu_old, self.softplus(self.rho_old), self.mu, self.softplus(self.rho))

    def kl_div_new_prior(self):
        return self.kl_div_p_q(self.mu, self.softplus(self.rho), 0., self.prior_sd)

    def kl_div_old_prior(self):
        return self.kl_div_p_q(self.mu_old, self.softplus(self.rho_old), 0., self.prior_sd)

    def kl_div_prior_new(self):
        return self.kl_div_p_q(0., self.prior_sd, self.mu, self.softplus(self.rho))

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        if self._matrix_variate_gaussian:
            # def transf(std):
            #     s_u = std[self.num_inputs:]
            #     s_v = std[:self.num_inputs]
            #     return T.dot(s_u.dimshuffle(0, 'x'), s_v.dimshuffle('x', 0)).T
            #
            # if not isinstance(p_std, float):
            #     p_std = transf(p_std)
            # if not isinstance(q_std, float):
            #     q_std = transf(q_std)
            return self.kl_div_p_q_mvg_full(p_mean, p_std, q_mean, q_std)
        else:
            numerator = T.square(p_mean - q_mean) + T.square(p_std) - T.square(q_std)
            denominator = 2 * T.square(q_std) + 1e-8
            return T.sum(numerator / denominator + T.log(q_std) - T.log(p_std))

    def kl_div_p_q_mvg_full(self, p_mean, p_std, q_mean, q_std):
        # Split rho's into different R1 matrices.
        lst_psu, lst_psv, lst_qsu, lst_qsv = [], [], [], []

        # import ipdb; ipdb.set_trace()

        def extract_uv(std, lst_su, lst_sv):
            for i in xrange(self.mvg_rank):
                su = std[
                     i * (self.num_inputs + self.num_units) + self.num_inputs:(i + 1) * (
                         self.num_inputs + self.num_units)]
                sv = std[i * (self.num_inputs + self.num_units): i * (
                    self.num_inputs + self.num_units) + self.num_inputs]
                lst_su.append(su)
                lst_sv.append(sv)

        extract_uv(p_std, lst_psu, lst_psv)
        if not isinstance(q_mean, float):
            extract_uv(q_std, lst_qsu, lst_qsv)

        def construct_matrix(lst_s):
            s = T.zeros((lst_s[0].shape[0], lst_s[0].shape[0]))
            for i in xrange(self.mvg_rank):
                _a = T.dot(lst_s[i].dimshuffle(0, 'x'), lst_s[i].dimshuffle('x', 0))
                s += _a
            return s

        if not isinstance(q_mean, float):
            qsu = construct_matrix(lst_qsu)
            qsv = construct_matrix(lst_qsv)
        import ipdb;
        ipdb.set_trace()
        psu = construct_matrix(lst_psu)
        psv = construct_matrix(lst_psv)

        # Sherman-Morrison
        def sherman_morrison(lst_s):
            A_inv = T.eye(lst_s[0].shape[0])
            lst_A_inv = [A_inv]
            for i in xrange(self.mvg_rank):
                _a = T.dot(A_inv, lst_s[i].dimshuffle(0, 'x'))
                _b = T.dot(_a, lst_s[i].dimshuffle('x', 0))
                _c = T.dot(_b, A_inv)
                A_inv -= _c
                lst_A_inv.append(A_inv)
            return A_inv, lst_A_inv

        if not isinstance(q_mean, float):
            qsu_inv, lst_qsu_inv = sherman_morrison(lst_qsu)
            qsv_inv, lst_qsv_inv = sherman_morrison(lst_qsv)
        import ipdb;
        ipdb.set_trace()
        psu_inv, lst_psu_inv = sherman_morrison(lst_psu)
        psv_inv, lst_psv_inv = sherman_morrison(lst_psv)

        # Calculate log determinant for rank1 updates.
        def log_determinant(lst_s, lst_s_inv):
            A_logdet = 1.
            for i in xrange(self.mvg_rank):
                _a = T.dot(lst_s[i].dimshuffle('x', 0), lst_s_inv[i])
                _b = T.dot(_a, lst_s[i].dimshuffle(0, 'x'))
                _c = 1 + _b
                A_logdet += _c
            return A_logdet

        if not isinstance(q_mean, float):
            qsu_logdet = log_determinant(lst_qsu, lst_qsu_inv)
            qsv_logdet = log_determinant(lst_qsv, lst_qsv_inv)
        import ipdb;
        ipdb.set_trace()
        psu_logdet = log_determinant(lst_psu, lst_psu_inv)
        psv_logdet = log_determinant(lst_psv, lst_psv_inv)

        if not isinstance(q_mean, float):
            _a = T.nlinalg.trace(T.dot(qsu_inv, psu))
            _b = T.nlinalg.trace(T.dot(qsv_inv, psv))
        else:
            _a = T.nlinalg.trace(psu)
            _b = T.nlinalg.trace(psv)
            qsu_logdet = 0
            qsv_logdet = 0
            qsv_inv = T.eye(lst_psv[0].shape[0])
            qsu_inv = T.eye(lst_psu[0].shape[0])

        _c = _a * _b
        _d = T.dot(q_mean - p_mean, qsu_inv)
        _e = T.dot(_d, (q_mean - p_mean).T)
        _f = T.dot(_e, qsv_inv)
        _g = T.nlinalg.trace(_f)

        _h = self.num_inputs * self.num_units
        _i = self.num_inputs * qsu_logdet
        _j = self.num_units * qsv_logdet
        _k = self.num_inputs * psu_logdet
        _l = self.num_units * psv_logdet

        kl = 0.5 * (_c + _g - _h + _i + _j - _k - _l)

        return kl


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
            activation = conved + self.get_b().dimshuffle(('x', 0) + ('x',) * self.n)

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
            return self.num_units,

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
                 **kwargs):
        super(BayesianDenseLayer, self).__init__(
            incoming, num_units, nonlinearity, prior_sd, **kwargs)

        self.use_local_reparametrization_trick = use_local_reparametrization_trick

        self.init_params()

    def get_W_shape(self):
        return self.num_inputs, self.num_units

    def get_b_shape(self):
        return self.num_units,

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

        if self._matrix_variate_gaussian:
            s = self.softplus(self.rho)
            s_u = s[self.num_inputs:]
            s_v = s[:self.num_inputs]
            s = T.dot(s_u.dimshuffle(0, 'x'), s_v.dimshuffle('x', 0)).dimshuffle(1, 0)
            # input \in M x num_inputs; s_u \in num_units x 1; s_v \in 1 x num_inputs
            # s \in num_inputs x num_units
            # out \in M x num_units
            delta = T.dot(T.square(input), T.square(s)) + T.mean(T.square(self.softplus(self.b_rho)))
        else:
            # input \in M x num_inputs; rho \in num_inputs x num_units; out \in M x num_units
            delta = T.dot(T.square(input), T.square(self.softplus(self.rho))) \
                    + T.square(self.softplus(self.b_rho)).dimshuffle('x', 0)

        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable

        activation = gamma + T.sqrt(delta) * epsilon * mask

        return self.nonlinearity(activation)

    def get_output_for(self, input, **kwargs):
        if self.use_local_reparametrization_trick and not self.disable_variance:
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
        return input_shape[0], self.num_units


if __name__ == '__main__':
    pass
