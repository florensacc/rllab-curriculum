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


class BNNLayer(lasagne.layers.Layer):
    """Probabilistic layer that uses Gaussian weights.

    Each weight has two parameters: mean and standard deviation (std).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 prior_sd=None,
                 group_variance_by=None,
                 use_local_reparametrization_trick=None,
                 disable_variance=None,
                 **kwargs):
        super(BNNLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        # Set vars.
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.prior_sd = prior_sd
        self.group_variance_by = group_variance_by
        self.use_local_reparametrization_trick = use_local_reparametrization_trick
        self.disable_variance = disable_variance

        # Convert prior_sd into prior_rho.
        prior_rho = self.std_to_log(self.prior_sd)

        self.W = np.random.normal(0., prior_sd,
                                  (self.num_inputs, self.num_units))  # @UndefinedVariable
        self.b = np.zeros(
            (self.num_units,),
            dtype=theano.config.floatX)  # @UndefinedVariable

        # Here we set the priors.
        # -----------------------
        self.mu = self.add_param(
            lasagne.init.Normal(1., 0.),
            (self.num_inputs, self.num_units),
            name='mu'
        )
        if self.group_variance_by == 'layer':
            self.rho = self.add_param(
                lasagne.init.Constant(prior_rho),
                (1, 1),
                name='rho',
                broadcastable=(True, True)
            )
        elif self.group_variance_by == 'unit':
            self.rho = self.add_param(
                lasagne.init.Constant(prior_rho),
                (self.num_inputs, ),
                name='rho',
                broadcastable=(False, True)
            )
        else:
            self.rho = self.add_param(
                lasagne.init.Constant(prior_rho),
                (self.num_inputs, self.num_units),
                name='rho'
            )
        # Bias priors.
        self.b_mu = self.add_param(
            lasagne.init.Normal(1., 0.),
            (self.num_units,),
            name="b_mu",
            regularizable=False
        )
        if self.group_variance_by == 'layer':
            self.b_rho = self.add_param(
                lasagne.init.Constant(prior_rho),
                (1,),
                name="b_rho",
                regularizable=False,
                broadcastable=(True,)
            )
        elif self.group_variance_by == 'unit':
            self.b_rho = self.add_param(
                lasagne.init.Constant(prior_rho),
                (1,),
                name="b_rho",
                regularizable=False
            )
        else:
            self.b_rho = self.add_param(
                lasagne.init.Constant(prior_rho),
                (self.num_units,),
                name="b_rho",
                regularizable=False
            )
            # -----------------------

        # Backup params for KL calculations.
        self.mu_old = self.add_param(
            np.zeros((self.num_inputs, self.num_units)),
            (self.num_inputs, self.num_units),
            name='mu_old',
            trainable=False,
            oldparam=True
        )
        if self.group_variance_by == 'layer':
            self.rho_old = self.add_param(
                np.ones((1, 1)),
                (1, 1),
                name='rho_old',
                trainable=False,
                oldparam=True,
                broadcastable=(True, True)
            )
        elif self.group_variance_by == 'unit':
            self.rho_old = self.add_param(
                np.ones((self.num_inputs, )),
                (self.num_inputs, ),
                name='rho_old',
                trainable=False,
                oldparam=True
            )
        else:
            self.rho_old = self.add_param(
                np.ones((self.num_inputs, self.num_units)),
                (self.num_inputs, self.num_units),
                name='rho_old',
                trainable=False,
                oldparam=True
            )
        # Bias priors.
        self.b_mu_old = self.add_param(
            np.zeros((self.num_units,)),
            (self.num_units,),
            name="b_mu_old",
            regularizable=False,
            trainable=False,
            oldparam=True
        )
        if self.group_variance_by == 'layer':
            self.b_rho_old = self.add_param(
                np.ones((1,)),
                (1,),
                name="b_rho_old",
                regularizable=False,
                trainable=False,
                oldparam=True
            )
        elif self.group_variance_by == 'unit':
            self.b_rho_old = self.add_param(
                np.ones((1,)),
                (1,),
                name="b_rho_old",
                regularizable=False,
                trainable=False,
                oldparam=True
            )
        else:
            self.b_rho_old = self.add_param(
                np.ones((self.num_units,)),
                (self.num_units,),
                name="b_rho_old",
                regularizable=False,
                trainable=False,
                oldparam=True
            )

    def log_to_std(self, rho):
        """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+

        This makes sure that we don't get negative stds. However, a downside might be
        that we have little gradient on close to 0 std (= -inf using this transformation).
        """
        return T.log(1 + T.exp(rho))

    def std_to_log(self, sigma):
        """Reverse log_to_std transformation."""
        return np.log(np.exp(sigma) - 1)

    def get_W(self):
        if self.disable_variance:
            mask = 0.
        else:
            mask = 1.
        if self.group_variance_by == 'layer':
            # Here we generate random epsilon values from a normal distribution
            epsilon = self._srng.normal(size=(1, 1), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable
            W = self.mu + T.mean(self.log_to_std(self.rho)) * epsilon * mask
        elif self.group_variance_by == 'unit':
            # Here we generate random epsilon values from a normal distribution
            epsilon = self._srng.normal(size=(self.num_inputs, ), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable
            W = self.mu + \
                (self.log_to_std(self.rho) * epsilon * mask).dimshuffle(0, 'x')
        elif self.group_variance_by == 'weight':
            # Here we generate random epsilon values from a normal distribution
            epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable
            W = self.mu + epsilon * self.log_to_std(self.rho) * mask
        else:
            raise Exception('Group variance by unknown!')
        self.W = W
        return W

    def get_b(self):
        if self.disable_variance:
            mask = 0.
        else:
            mask = 1.
        if self.group_variance_by == 'layer':
            # Here we generate random epsilon values from a normal distribution
            epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable
            # T.mean is a hack to get 2D broadcasting on a scalar.
            b = self.b_mu + \
                T.mean(self.log_to_std(self.b_rho)) * epsilon * mask
        elif self.group_variance_by == 'unit':
            # Here we generate random epsilon values from a normal distribution
            epsilon = self._srng.normal(size=(1,), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable
            b = self.b_mu + \
                T.mean(self.log_to_std(self.b_rho)) * epsilon * mask
        elif self.group_variance_by == 'weight':
            # Here we generate random epsilon values from a normal distribution
            epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable
            b = self.b_mu + self.log_to_std(self.b_rho) * epsilon * mask
        else:
            raise Exception('Group variance by unknown!')
        self.b = b
        return b

    def get_output_for_reparametrization(self, input, **kwargs):
        """Implementation of the local reparametrization trick.

        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.

        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """

        if self.group_variance_by == 'layer':
            raise Exception(
                'Local reparametrization trick not supported for tied variances per layer!')
        else:
            if input.ndim > 2:
                # if the input has more than two dimensions, flatten it into a
                # batch of feature vectors.
                input = input.flatten(2)

            gamma = T.dot(input, self.mu) + self.b_mu.dimshuffle('x', 0)
            delta = T.dot(T.square(input), T.square(self.log_to_std(
                self.rho))) + T.square(self.log_to_std(self.b_rho)).dimshuffle('x', 0)

            if self.disable_variance:
                mask = 0.
            else:
                mask = 1.
            epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable

            activation = gamma + T.sqrt(delta) * epsilon * mask

            return self.nonlinearity(activation)

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

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        if self.group_variance_by == 'layer':
            p_std = T.mean(p_std)
            q_std = T.mean(q_std)
        elif self.group_variance_by == 'unit':
            if not isinstance(p_std, float):
                p_std = p_std.dimshuffle(0, 'x')
            if not isinstance(q_std, float):
                q_std = q_std.dimshuffle(0, 'x')
        numerator = T.square(p_mean - q_mean) + \
            T.square(p_std) - T.square(q_std)
        denominator = 2 * T.square(q_std) + 1e-8
        return T.sum(
            numerator / denominator + T.log(q_std) - T.log(p_std))

    def kl_div_new_old(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), self.mu_old, self.log_to_std(self.rho_old))
        kl_div += self.kl_div_p_q(self.b_mu, self.log_to_std(self.b_rho),
                                  self.b_mu_old, self.log_to_std(self.b_rho_old))
        return kl_div

    def kl_div_old_new(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), self.mu, self.log_to_std(self.rho))
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  self.log_to_std(self.b_rho_old), self.b_mu, self.log_to_std(self.b_rho))
        return kl_div

    def kl_div_new_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu, self.log_to_std(self.rho), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu,
                                  self.log_to_std(self.b_rho), 0., self.prior_sd)
        return kl_div

    def kl_div_old_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, self.log_to_std(self.rho_old), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  self.log_to_std(self.b_rho_old), 0., self.prior_sd)
        return kl_div

    def kl_div_prior_new(self):
        kl_div = self.kl_div_p_q(
            0., self.prior_sd, self.mu,  self.log_to_std(self.rho))
        kl_div += self.kl_div_p_q(0., self.prior_sd,
                                  self.b_mu, self.log_to_std(self.b_rho))
        return kl_div

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


class CatOutBNNLayer(BNNLayer):
    """ Categorical output layer (multidimensional softmax); extension to BNNLayer """

    def __init__(self,
                 incoming,
                 num_units,
                 nonlinearity=lasagne.nonlinearities.softmax,
                 prior_sd=None,
                 group_variance_by=None,
                 use_local_reparametrization_trick=None,
                 num_classes=None,
                 num_output_dim=None,
                 disable_variance=None,
                 **kwargs):
        super(CatOutBNNLayer, self).__init__(incoming, num_units, nonlinearity,
                                             prior_sd, group_variance_by, use_local_reparametrization_trick, disable_variance, **kwargs)
        self.num_classes = num_classes
        self.num_output_dim = num_output_dim

    def get_output_for_default(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.get_W()) + \
            self.get_b().dimshuffle('x', 0)

        # Apply nonlinearity (softmax) over all n_out dimensions.
        postact = self.nonlinearity(activation.reshape([-1, self.num_classes]))
        return postact.reshape([-1, self.num_classes * self.num_output_dim])

    def get_output_for_reparametrization(self, input, **kwargs):
        """Implementation of the local reparametrization trick.

        This essentially leads to a speedup compared to the naive implementation case.
        Furthermore, it leads to gradients with less variance.

        References
        ----------
        Kingma et al., "Variational Dropout and the Local Reparametrization Trick", 2015
        """

        if self.group_variance_by == 'layer':
            raise Exception(
                'Local reparametrization trick not supported for tied variances per layer!')
        else:
            if input.ndim > 2:
                # if the input has more than two dimensions, flatten it into a
                # batch of feature vectors.
                input = input.flatten(2)

            gamma = T.dot(input, self.mu) + self.b_mu.dimshuffle('x', 0)
            delta = T.dot(T.square(input), T.square(self.log_to_std(
                self.rho))) + T.square(self.log_to_std(self.b_rho)).dimshuffle('x', 0)

            if self.disable_variance:
                mask = 0.
            else:
                mask = 1.
            epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                        dtype=theano.config.floatX)  # @UndefinedVariable

            activation = gamma + T.sqrt(delta) * epsilon * mask

        # Apply nonlinearity (softmax) over all n_out dimensions.
        postact = self.nonlinearity(activation.reshape([-1, self.num_classes]))
        return postact.reshape([-1, self.num_classes * self.num_output_dim])


class BNN(LasagnePowered, Serializable):
    """Bayesian neural network (BNN), according to Blundell2016."""

    def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 layers_type,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.5,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 second_order_update=False,
                 learning_rate=0.0001,
                 compression=False,
                 information_gain=True,
                 update_prior=False,
                 update_likelihood_sd=False,
                 group_variance_by='weight',
                 use_local_reparametrization_trick=True,
                 likelihood_sd_init=1.0,
                 output_type='regression',
                 num_classes=None,
                 num_output_dim=None,
                 disable_variance=False,
                 debug=False
                 ):

        Serializable.quick_init(self, locals())
        assert len(layers_type) == len(n_hidden) + 1

        assert group_variance_by in ['layer', 'unit', 'weight']

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_type = layers_type
        self.n_batches = n_batches
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.likelihood_sd_init = likelihood_sd_init
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.compression = compression
        self.information_gain = information_gain
        self.update_prior = update_prior
        self.update_likelihood_sd = update_likelihood_sd
        self.group_variance_by = group_variance_by
        self.use_local_reparametrization_trick = use_local_reparametrization_trick
        self.output_type = output_type
        self.num_classes = num_classes
        self.num_output_dim = num_output_dim
        self.disable_variance = disable_variance
        self.debug = debug

        if self.output_type == 'classification':
            assert self.num_classes is not None
            assert self.num_output_dim is not None
            assert self.n_out == self.num_classes * self.num_output_dim

        if self.group_variance_by != 'weight':
            assert not self.use_local_reparametrization_trick

        if self.output_type == 'classification' and self.update_likelihood_sd:
            print(
                'Setting output_type=\'classification\' cannot be used with update_likelihood_sd=True, changing to False.')
            self.update_likelihood_sd = False

        if self.disable_variance:
            print('Warning: all noise has been disabled, only using means.')

        assert self.information_gain or self.compression

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()

    def save_old_params(self):
        layers = filter(lambda l: isinstance(l, BNNLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.save_old_params()
        if self.update_likelihood_sd:
            self.old_likelihood_sd.set_value(self.likelihood_sd.get_value())

    def reset_to_old_params(self):
        layers = filter(lambda l: isinstance(l, BNNLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.reset_to_old_params()
        if self.update_likelihood_sd:
            self.likelihood_sd.set_value(self.old_likelihood_sd.get_value())

    def compression_improvement(self):
        """KL divergence KL[old_param||new_param]"""
        layers = filter(lambda l: isinstance(l, BNNLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BNNLayer),
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
        layers = filter(lambda l: isinstance(l, BNNLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: isinstance(l, BNNLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = filter(lambda l: isinstance(l, BNNLayer),
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
        # return T.sum(T.log(prediction[T.arange(target.shape[0]), target]))

        # Cross-entropy; target vector selecting correct prediction
        # entries.
        # Hardcoding n_classes=3
        target2 = target + T.arange(target.shape[1]) * self.num_classes
        target3 = target2.T.ravel()
        idx = T.arange(target.shape[0])
        idx2 = T.tile(idx, self.num_output_dim)
        prediction_selected = prediction[
            idx2, target3].reshape([self.num_output_dim, target.shape[0]]).T
        ll = T.sum(T.log(prediction_selected))
        return ll

    def loss(self, input, target, kl_factor, **kwargs):

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.output_type == 'classification':
                lh = self.likelihood_classification(target, prediction)
            elif self.output_type == 'regression':
                lh = self.likelihood_regression(target, prediction, **kwargs)
            _log_p_D_given_w.append(lh)
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
        return kl / self.n_batches * kl_factor - log_p_D_given_w / self.n_samples

    def loss_last_sample(self, input, target, **kwargs):
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
            if self.output_type == 'classification':
                lh = self.likelihood_classification(target, prediction)
            elif self.output_type == 'regression':
                lh = self.likelihood_regression(target, prediction, **kwargs)
            _log_p_D_given_w.append(lh)
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

        # Input layer
        network = lasagne.layers.InputLayer(shape=(1, self.n_in))

        # Hidden layers
        for i in xrange(len(self.n_hidden)):
            if self.layers_type[i] == 'gaussian':
                network = BNNLayer(
                    network, self.n_hidden[
                        i], nonlinearity=self.transf, prior_sd=self.prior_sd, group_variance_by=self.group_variance_by,
                    disable_variance=self.disable_variance, use_local_reparametrization_trick=self.use_local_reparametrization_trick)
            elif self.layers_type[i] == 'deterministic':
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf)

        # Output layer
        if self.output_type == 'regression':
            if self.layers_type[len(self.n_hidden)] == 'gaussian':
                network = BNNLayer(
                    network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, group_variance_by=self.group_variance_by,
                    disable_variance=self.disable_variance, use_local_reparametrization_trick=self.use_local_reparametrization_trick)
            elif self.layers_type[len(self.n_hidden)] == 'deterministic':
                network = lasagne.layers.DenseLayer(
                    network, self.n_out, nonlinearity=self.outf)
        elif self.output_type == 'classification':
            network = CatOutBNNLayer(
                network, self.n_out, nonlinearity=lasagne.nonlinearities.softmax,
                prior_sd=self.prior_sd, group_variance_by=self.group_variance_by,
                num_classes=self.num_classes, num_output_dim=self.num_output_dim, disable_variance=self.disable_variance,
                use_local_reparametrization_trick=self.use_local_reparametrization_trick)

        self.network = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        kl_factor = T.scalar('kl_factor',
                             dtype=theano.config.floatX)  # @UndefinedVariable
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)  # @UndefinedVariable
        if self.output_type == 'regression':

            target_var = T.matrix('targets',
                                  dtype=theano.config.floatX)  # @UndefinedVariable

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

        elif self.output_type == 'classification':

            target_var = T.imatrix('targets')

            # Loss function.
            loss = self.loss(
                input_var, target_var, kl_factor)
            loss_only_last_sample = self.loss_last_sample(
                input_var, target_var)

        # Create update methods.
        params_kl = lasagne.layers.get_all_params(self.network, trainable=True)
        params = []
        params.extend(params_kl)
        if self.update_likelihood_sd:
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

            if self.debug:

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
                        elif param.name == 'likelihood_sd':
                            invH = 0.
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
                        elif param.name == 'likelihood_sd':
                            invH = 0.
                        invHs.append(invH)
                    return grads

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

            if self.debug:
                self.debug_H = ext.compile_function(
                    [input_var, target_var], debug_H(
                        loss_only_last_sample, params, oldparams),
                    log_name='fn_debug_grads')
                self.debug_g = ext.compile_function(
                    [input_var, target_var], debug_g(
                        loss_only_last_sample, params, oldparams),
                    log_name='fn_debug_grads')

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

        if self.debug:
            self.eval_loss = ext.compile_function(
                [input_var, target_var, kl_factor], loss,  log_name='fn_eval_loss', no_default_updates=False)
            # Calculate surprise.
            self.fn_surprise = ext.compile_function(
                [], self.surprise(), log_name='fn_surprise')
#             self.fn_dbg_nll = ext.compile_function(
#                 [input_var, target_var], self.dbg_nll(input_var, target_var, self.likelihood_sd), log_name='fn_dbg_nll', no_default_updates=False)
            self.fn_kl = ext.compile_function(
                [], self.kl_div(), log_name='fn_kl')
            self.fn_kl_from_prior = ext.compile_function(
                [], self.log_p_w_q_w_kl(), log_name='fn_kl_from_prior')
            self.fn_classification_nll = ext.compile_function(
                [input_var, target_var], self.likelihood_classification(target_var, self.pred_sym(input_var)), log_name='fn_classification_nll')

if __name__ == '__main__':
    pass
