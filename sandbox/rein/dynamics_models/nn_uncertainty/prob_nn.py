from __future__ import print_function
import time
import numpy as np
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from sandbox.rein.dynamics_models.bnn.utils import sliding_mean, iterate_minibatches
import theano

# Plotting params.
# ----------------
# Only works for regression
PLOT_WEIGHTS_INDIVIDUAL = False
PLOT_WEIGHTS_TOTAL = False
PLOT_OUTPUT = True
PLOT_OUTPUT_REGIONS = False
PLOT_KL = False
# ----------------
VBNN_LAYER_TAG = 'unnlayer'
USE_REPARAMETRIZATION_TRICK = True


def log_to_std(rho):
    """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+

    This makes sure that we don't get negative stds. However, a downside might be
    that we have little gradient on close to 0 std (= -inf using this transformation).
    """
    return T.log(1 + T.exp(rho))


class ProbLayer(lasagne.layers.Layer):
    """Probabilistic layer that uses Gaussian weights.

    Each weight has two parameters: mean and standard deviation (std).
    """

    def __init__(self,
                 incoming,
                 num_units,
                 mu=lasagne.init.Normal(std=1., mean=0.),
                 rho=lasagne.init.Normal(std=1., mean=0.),
                 b_mu=lasagne.init.Normal(std=1., mean=0.),
                 b_rho=lasagne.init.Normal(std=1., mean=0.),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 prior_sd=None,
                 **kwargs):
        super(ProbLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        # Set vars.
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.prior_sd = prior_sd

        self.W = np.zeros(
            (self.num_inputs, self.num_units),
            dtype=theano.config.floatX)  # @UndefinedVariable
        self.b = np.zeros(
            (self.num_units,),
            dtype=theano.config.floatX)  # @UndefinedVariable

        # Here we set the priors.
        self.mu = self.add_param(
            mu, (self.num_inputs, self.num_units), name='mu')
        self.rho = self.add_param(
            rho, (self.num_inputs, self.num_units), name='rho')
        # Bias priors.
        self.b_mu = self.add_param(b_mu, (self.num_units,), name="b_mu",
                                   regularizable=False)
        self.b_rho = self.add_param(b_rho, (self.num_units,), name="b_rho",
                                    regularizable=False)

        # Backup params for KL calculations.
        self.mu_old = self.add_param(
            mu, (self.num_inputs, self.num_units), name='mu_old', trainable=False)
        self.rho_old = self.add_param(
            rho, (self.num_inputs, self.num_units), name='rho_old', trainable=False)
        # Bias priors.
        self.b_mu_old = self.add_param(b_mu, (self.num_units,), name="b_mu_old",
                                       regularizable=False, trainable=False)
        self.b_rho_old = self.add_param(b_rho, (self.num_units,), name="b_rho_old",
                                        regularizable=False, trainable=False)

    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        # (paper step 1)
        epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=self.prior_sd,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + log_to_std(self.rho) * epsilon
        self.W = W
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        # (paper step 1)
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=self.prior_sd,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        b = self.b_mu + log_to_std(self.b_rho) * epsilon
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
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        gamma = T.dot(input, self.mu) + self.b_mu.dimshuffle('x', 0)
        delta = T.dot(T.square(input), T.square(log_to_std(
            self.rho))) + T.square(log_to_std(self.b_rho)).dimshuffle('x', 0)
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=self.prior_sd,
                                    dtype=theano.config.floatX)  # @UndefinedVariable

        activation = gamma + T.sqrt(delta) * epsilon

        return self.nonlinearity(activation)

    def get_output_for_default(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.get_W()) + \
            self.get_b().dimshuffle('x', 0)

        return self.nonlinearity(activation)

    def get_output_for(self, input, **kwargs):
        if USE_REPARAMETRIZATION_TRICK:
            return self.get_output_for_reparametrization(input, **kwargs)
        else:
            return self.get_output_for_default(input, **kwargs)

    def save_old_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old.set_value(self.mu.get_value())
        self.rho_old.set_value(self.rho.get_value())
        self.b_mu_old.set_value(self.b_mu.get_value())
        self.b_rho_old.set_value(self.b_rho.get_value())

    def reset_to_old_params(self):
        self.mu.set_value(self.mu_old.get_value())
        self.rho.set_value(self.rho_old.get_value())
        self.b_mu.set_value(self.b_mu_old.get_value())
        self.b_rho.set_value(self.b_rho_old.get_value())

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        numerator = T.square(p_mean - q_mean) + \
            T.square(p_std) - T.square(q_std)
        denominator = 2 * T.square(q_std) + 1e-8
        return T.sum(
            numerator / denominator + T.log(q_std) - T.log(p_std))

    def kl_div_new_old(self):
        old_mean = self.mu_old
        old_std = log_to_std(self.rho_old) * self.prior_sd
        new_mean = self.mu
        new_std = log_to_std(self.rho) * self.prior_sd
        kl_div = self.kl_div_p_q(new_mean, new_std, old_mean, old_std)

        old_mean = self.b_mu_old
        old_std = log_to_std(self.b_rho_old) * self.prior_sd
        new_mean = self.b_mu
        new_std = log_to_std(self.b_rho) * self.prior_sd
        kl_div += self.kl_div_p_q(new_mean, new_std, old_mean, old_std)

        return kl_div

    def kl_div_old_new(self):
        old_mean = self.mu_old
        old_std = log_to_std(self.rho_old) * self.prior_sd
        new_mean = self.mu
        new_std = log_to_std(self.rho) * self.prior_sd
        kl_div = self.kl_div_p_q(old_mean, old_std, new_mean, new_std)

        old_mean = self.b_mu_old
        old_std = log_to_std(self.b_rho_old) * self.prior_sd
        new_mean = self.b_mu
        new_std = log_to_std(self.b_rho) * self.prior_sd
        kl_div += self.kl_div_p_q(old_mean, old_std, new_mean, new_std)

        return kl_div

    def kl_div_new_prior(self):
        prior_mean = 0.
        prior_std = self.prior_sd
        new_mean = self.mu
        new_std = log_to_std(self.rho) * self.prior_sd
        kl_div = self.kl_div_p_q(new_mean, new_std, prior_mean, prior_std)

        new_mean = self.b_mu
        new_std = log_to_std(self.b_rho) * self.prior_sd
        kl_div += self.kl_div_p_q(new_mean, new_std, prior_mean, prior_std)

        return kl_div

    def kl_div_prior_new(self):
        prior_mean = 0.
        prior_std = self.prior_sd
        new_mean = self.mu
        new_std = log_to_std(self.rho) * self.prior_sd
        kl_div = self.kl_div_p_q(prior_mean, prior_std, new_mean, new_std)

        new_mean = self.b_mu
        new_std = log_to_std(self.b_rho) * self.prior_sd
        kl_div += self.kl_div_p_q(prior_mean, prior_std, new_mean, new_std)

        return kl_div

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class ProbNN:
    """Neural network with weight uncertainty

    """

    def __init__(self, n_in,
                 n_hidden,
                 n_out,
                 layers_type,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 batch_size=100,
                 n_samples=10,
                 prior_sd=0.05,
                 type='regression',
                 reverse_update_kl=False,
                 symbolic_prior_kl=True,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 stochastic_output=False
                 ):

        self._srng = RandomStreams()
        assert len(layers_type) == len(n_hidden) + 1

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.prior_sd = prior_sd
        self.layers_type = layers_type
        self.type = type
        self.n_batches = n_batches
        self.reverse_update_kl = reverse_update_kl
        self.symbolic_prior_kl = symbolic_prior_kl
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        if self.use_reverse_kl_reg:
            assert self.symbolic_prior_kl == True
        self.stochastic_output = stochastic_output

    def _get_prob_layers(self):
        if self.stochastic_output:
            layers_mean = filter(lambda l: l.name == VBNN_LAYER_TAG,
                                 lasagne.layers.get_all_layers(self.network_mean)[1:])
            layers_stdn = filter(lambda l: l.name == VBNN_LAYER_TAG,
                                 lasagne.layers.get_all_layers(self.network_stdn)[1:])
            layers = layers_mean + layers_stdn
        else:
            layers = filter(lambda l: l.name == VBNN_LAYER_TAG,
                            lasagne.layers.get_all_layers(self.network)[1:])
        return layers

    def save_old_params(self):
        for layer in self._get_prob_layers():
            layer.save_old_params()

    def reset_to_old_params(self):
        for layer in self._get_prob_layers():
            layer.reset_to_old_params()

    def get_kl_div_sampled(self):
        """Sampled KL calculation"""
        kl_div = 0.
        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        for layer in self._get_prob_layers():
            if layer.name == VBNN_LAYER_TAG:
                W = layer.get_W()
                b = layer.get_b()
                kl_div += self._log_prob_normal(W,
                                                layer.mu, T.log(1 + T.exp(layer.rho)) * self.prior_sd)
                kl_div += self._log_prob_normal(b,
                                                layer.b_mu, T.log(1 + T.exp(layer.b_rho)) * self.prior_sd)
                kl_div -= self._log_prob_normal(W,
                                                layer.mu_old, T.log(1 + T.exp(layer.rho_old)) * self.prior_sd)
                kl_div -= self._log_prob_normal(b,
                                                layer.b_mu_old, T.log(1 + T.exp(layer.b_rho_old)) * self.prior_sd)
        return kl_div

    def rev_kl_div(self):
        """KL divergence KL[old_param||new_param]"""
        return sum(l.kl_div_old_new() for l in self._get_prob_layers())

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        return sum(l.kl_div_new_old() for l in self._get_prob_layers())

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        return sum(l.kl_div_new_prior() for l in self._get_prob_layers())

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        return sum(l.kl_div_prior_new() for l in self._get_prob_layers())

    def _log_prob_normal(self, input, mu=0., sigma=0.01):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)

    def _log_prob_spike_and_slab(self, input, pi, mu, sigma0, sigma1):
        """sigma0 > sigma1"""
        log_prob = pi * self._log_prob_normal(input, mu=mu, sigma=sigma0) + (
            1 - pi) * self._log_prob_normal(input, mu=mu, sigma=sigma1)
        return log_prob

    def pred_sym_default(self, input):
        """Default output"""
        return lasagne.layers.get_output(self.network, input)

    # For stochastic output
    # ---------------------
    def pred_mean(self, input):
        return lasagne.layers.get_output(self.network_mean, input)

    def pred_stdn(self, input):
        return log_to_std(lasagne.layers.get_output(self.network_stdn, input))
    # ---------------------

    def pred_sym_stochastic(self, input):
        """Gaussian output sampled."""
        # Mean is a matrix of batch_size rows.
        mean = self.pred_mean(input)
        stdn = self.pred_stdn(input)
        epsilon = self._srng.normal(size=(1,), avg=0., std=1.)
        out = mean + epsilon * stdn
        return out

    def pred_sym(self, input):
        if self.stochastic_output:
            return self.pred_sym_stochastic(input)
        else:
            return self.pred_sym_default(input)

    def get_loss_sym_sym(self, input, target):

        log_p_D_given_w = 0.

        # MC samples.
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.type == 'regression':
                if self.stochastic_output:
                    log_p_D_given_w += self._log_prob_normal(
                        target, self.pred_mean(input), self.pred_stdn(input))
                else:
                    log_p_D_given_w += self._log_prob_normal(
                        target, prediction, self.prior_sd)
            elif self.type == 'classification':
                log_p_D_given_w += T.sum(
                    T.log(prediction)[T.arange(target.shape[0]), T.cast(target[:, 0], dtype='int64')])

        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.log_p_w_q_w_kl()
        if self.use_reverse_kl_reg:
            kl += self.reverse_kl_reg_factor * \
                self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
        loss = (kl / self.n_batches -
                log_p_D_given_w) / self.batch_size
        loss /= self.n_samples

        return loss

    def get_loss_only_last_sample(self, input, target):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """

        log_p_D_given_w = 0.

        # MC samples.
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.type == 'regression':
                if self.stochastic_output:
                    log_p_D_given_w += self._log_prob_normal(
                        target, self.pred_mean(input), self.pred_stdn(input))
                else:
                    log_p_D_given_w += self._log_prob_normal(
                        target, prediction, self.prior_sd)
            elif self.type == 'classification':
                log_p_D_given_w += T.sum(
                    T.log(prediction)[T.arange(target.shape[0]), T.cast(target[:, 0], dtype='int64')])

        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.kl_div()
        if self.use_reverse_kl_reg:
            kl += self.reverse_kl_reg_factor * \
                self.reverse_log_p_w_q_w_kl()

        # Calculate loss function.
        loss = (kl / self.n_batches -
                log_p_D_given_w) / self.batch_size
        loss /= self.n_samples

        return loss

    def get_loss_sym(self, input, target):
        raise Exception("Deprecated")

        log_q_w, log_p_w, log_p_D_given_w = 0., 0., 0.

        # MC samples.
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.type == 'regression':
                if self.stochastic_output:
                    log_p_D_given_w += self._log_prob_normal(
                        target, self.pred_mean(input), self.pred_stdn(input))
                else:
                    log_p_D_given_w += self._log_prob_normal(
                        target, prediction, self.prior_sd)
            elif self.type == 'classification':
                log_p_D_given_w += T.sum(
                    T.log(prediction)[T.arange(target.shape[0]), T.cast(target[:, 0], dtype='int64')])
            # Calculate variational posterior log(q(w)) and prior log(p(w)).
            layers = lasagne.layers.get_all_layers(self.network)[1:]
            for layer in layers:
                if layer.name == VBNN_LAYER_TAG:
                    W = layer.W
                    b = layer.b
                    log_q_w += self._log_prob_normal(W,
                                                     layer.mu, T.log(1 + T.exp(layer.rho)) * self.prior_sd)
                    log_q_w += self._log_prob_normal(b,
                                                     layer.b_mu, T.log(1 + T.exp(layer.b_rho)) * self.prior_sd)
                    log_p_w += self._log_prob_normal(W, 0., self.prior_sd)
                    log_p_w += self._log_prob_normal(b, 0., self.prior_sd)

        # Calculate loss function.
        loss = ((log_q_w - log_p_w) / self.n_batches -
                log_p_D_given_w) / self.batch_size
        loss /= self.n_samples

        return loss

    def build_network(self):

        # Input layer
        input = lasagne.layers.InputLayer(shape=(self.batch_size, self.n_in))

        # Hidden layers
        network = input
        for i in xrange(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            if self.layers_type[i] == 1:
                network = ProbLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, prior_sd=self.prior_sd, name=VBNN_LAYER_TAG)
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf)

        if self.stochastic_output:
            # Gaussian output
            # ---------------
            # Mean output subnet: actual network
            if self.layers_type[len(self.n_hidden)] == 1:
                network = ProbLayer(
                    network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name=VBNN_LAYER_TAG)
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_out, nonlinearity=self.outf)

            self.network_mean = network

            # Stdn output subnet: this connects directly to input
            network = input
            if self.layers_type[len(self.n_hidden)] == 1:
                network = ProbLayer(
                    network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name=VBNN_LAYER_TAG)
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_out, nonlinearity=self.outf)

            self.network_stdn = network
            # ---------------

        else:
            # Nonstochastic output
            # --------------------
            if self.layers_type[len(self.n_hidden)] == 1:
                network = ProbLayer(
                    network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name=VBNN_LAYER_TAG)
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_out, nonlinearity=self.outf)
            self.network = network
            # --------------------

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)  # @UndefinedVariable
        target_var = T.matrix('targets',
                              dtype=theano.config.floatX)  # @UndefinedVariable

        # Loss function.
        loss = self.get_loss_sym_sym(
            input_var, target_var)
        loss_only_last_sample = self.get_loss_only_last_sample(
            input_var, target_var)

        # Create update methods.
        if self.stochastic_output:
            params_mean = lasagne.layers.get_all_params(
                self.network_mean, trainable=True)
            params_stnd = lasagne.layers.get_all_params(
                self.network_stdn, trainable=True)
            params = params_mean + params_stnd
        else:
            params = lasagne.layers.get_all_params(
                self.network, trainable=True)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=0.001)

        # Train/val fn.
        self.pred_fn = theano.function(
            [input_var], self.pred_sym(input_var), allow_input_downcast=True)
        self.train_fn = theano.function(
            [input_var, target_var], loss, updates=updates, allow_input_downcast=True)
        self.train_update_fn = theano.function(
            [input_var, target_var], loss_only_last_sample, updates=updates, allow_input_downcast=True)

        self.train_err_fn = theano.function(
            [input_var, target_var], loss, allow_input_downcast=True)
        if self.reverse_update_kl:
            self.f_kl_div_closed_form = theano.function(
                [], self.get_kl_div_closed_form_reversed(), allow_input_downcast=True)
        else:
            self.f_kl_div_closed_form = theano.function(
                [], self.kl_div(), allow_input_downcast=True)

    def train(self, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):

        training_data_start = 1000
        training_data_end = 1100

        print('Training ...')

        kl_div_means = []
        kl_div_stdns = []
        kl_all_values = []

        # Plotting
        # ---------------------

        # Print weights from this layer.
#         layer = lasagne.layers.get_all_layers(self.network)[-1]

        if PLOT_WEIGHTS_TOTAL:
            import matplotlib.pyplot as plt
            plt.ion()
            sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
            mean = layer.mu.eval().ravel()
            painter_weights_total, = plt.plot(
                mean, sd, 'o', color=(1.0, 0, 0, 0.5))
            plt.xlim(xmin=-10, xmax=10)
            plt.ylim(ymin=0, ymax=5)
            plt.draw()
            plt.show()

        elif PLOT_WEIGHTS_INDIVIDUAL:
            def normal(x, mean, sd):
                return 1. / (sd * np.sqrt(2 * np.pi)) * np.exp(-(x - mean)**2 / (2 * sd**2))
            import matplotlib.pyplot as plt
            plt.ion()
            n_plots_h = 3
            n_plots_v = 3
            n_plots = n_plots_h * n_plots_v
            plt.ion()
            x = np.linspace(-6, 6, 100)
            _f, axarr = plt.subplots(
                n_plots_v, n_plots_h, sharex=True, sharey=True)
            painter_weights_individual = []
            for i in xrange(n_plots_v):
                for j in xrange(n_plots_h):
                    hl, = axarr[i][j].plot(x, x)
                    axarr[i][j].set_ylim(ymin=0, ymax=2)
                    painter_weights_individual.append(hl)
            plt.draw()
            plt.show()

        elif PLOT_OUTPUT:
            import matplotlib.pyplot as plt
            plt.ion()
            y = [self.pred_fn(x[None, :])[0][0] for x in X_test]
            _ = plt.figure()
            plt.plot(np.array(X_test[training_data_start:training_data_end])[:, 0][:, None], np.array(
                T_test[training_data_start:training_data_end]), 'o', label="t", color=(1.0, 0, 0, 0.5))
            painter_output, = plt.plot(np.array(X_test)[:, 0][:, None], np.array(
                y), 'o', label="y", color=(0, 0.7, 0, 0.2))
            plt.xlim(xmin=-7, xmax=8)
            plt.ylim(ymin=-4, ymax=4)
#             plt.xlim(xmin=-1.5, xmax=2.5)
#             plt.ylim(ymin=0, ymax=2)
            plt.draw()
            # For Jupyter notebook
            try:
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
            except:
                pass
            plt.show()
            plt.pause(0.0001)

        elif PLOT_KL:
            import matplotlib.pyplot as plt
            plt.ion()
            _, ax = plt.subplots(1)
            painter_kl, = ax.plot([], [], label="y", color=(1, 0, 0, 1))
            plt.xlim(xmin=0 * self.n_batches, xmax=100)
            plt.ylim(ymin=0, ymax=0.1)
            ax.grid()
            plt.draw()
            plt.show()
        # ---------------------

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

            # Iterate over all minibatches and train on each of them.
            for batch in iterate_minibatches(X_train, T_train, self.batch_size, shuffle=True):

                # Fix old params for KL divergence computation.
                self.save_old_params()

                # Train current minibatch.
                inputs, targets = batch
                _train_err = self.train_fn(inputs, targets)
                train_err += _train_err
                train_batches += 1

                # Calculate current minibatch KL.
                kl_mb_closed_form = self.f_kl_div_closed_form()

#                 a = self.calc_gradients(inputs, targets)
#                 print(len(a), len(a[0]))

                kl_values.append(kl_mb_closed_form)
                kl_all_values.append(kl_mb_closed_form)

            # Calculate KL divergence variance over all minibatches.
            kl_mean = np.mean(np.asarray(kl_values))
            kl_stdn = np.std(np.asarray(kl_values))

            kl_div_means.append(kl_mean)
            kl_div_stdns.append(kl_stdn)

            ##########################
            ### PLOT FUNCTIONALITY ###
            ##########################
            if PLOT_WEIGHTS_TOTAL:
                sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
                mean = layer.mu.eval().ravel()
                painter_weights_total.set_xdata(mean)
                painter_weights_total.set_ydata(sd)
                plt.draw()

            elif PLOT_WEIGHTS_INDIVIDUAL:
                for i in xrange(n_plots):
                    w_mu = layer.mu.eval()[i, 0]
                    w_rho = layer.rho.eval()[i, 0]
                    w_sigma = np.log(1 + np.exp(w_rho))
                    y = normal(x, w_mu, w_sigma)
                    painter_weights_individual[i].set_ydata(y)
                plt.draw()

            elif PLOT_OUTPUT:
                y = [self.pred_fn(x[None, :])[0][0] for x in X_test]
                painter_output.set_ydata(y)
                plt.draw()
                plt.pause(0.0001)

            elif PLOT_OUTPUT_REGIONS and epoch % 30 == 0 and epoch != 0:
                import matplotlib.pyplot as plt

                ys = []
                for i in xrange(100):
                    y = [self.pred_fn(x[None, :])[0][0] for x in X_test]
                    y = np.asarray(y)[:, None]
                    ys.append(y)
                ys = np.hstack(ys)
                y_mean = np.mean(ys, axis=1)
                y_std = np.std(ys, axis=1)
                y_median = np.median(ys, axis=1)
                y_first_quart = np.percentile(ys, q=25, axis=1)
                y_third_quart = np.percentile(ys, q=75, axis=1)
                indices = [i[0]
                           for i in sorted(enumerate(X_test[:, 0][:, None]), key=lambda x:x[1])]
                y_mean = y_mean[indices].flatten()
                y_std = y_std[indices].flatten()
                y_median = y_median[indices].flatten()
                y_first_quart = y_first_quart[indices].flatten()
                y_third_quart = y_third_quart[indices].flatten()
                _X_test = np.array(X_test[indices][:, 0][:, None]).flatten()

                window_size = 25
                y_mean = sliding_mean(y_mean,
                                      window=window_size)
                y_std = sliding_mean(y_std,
                                     window=window_size)

                _, axarr = plt.subplots(2, figsize=(16, 9))
                axarr[0].set_title('output')
                axarr[1].set_title('std')
                axarr[0].plot(np.array(X_test[training_data_start:training_data_end])[:, 0][:, None], np.array(
                    T_test[training_data_start:training_data_end]), 'o', label="t",
                    color=(1.0, 0, 0, 0.5))
                axarr[0].fill_between(
                    _X_test, (y_mean - y_std), (y_mean + y_std), interpolate=True, color=(0, 0, 0, 0.2))
                axarr[0].fill_between(
                    _X_test, (y_mean - 2 * y_std), (y_mean + 2 * y_std), interpolate=True, color=(0, 0, 0, 0.2))
                axarr[0].plot(
                    _X_test, y_mean)
                axarr[1].plot(
                    _X_test, y_std)
                rnd_indices = np.random.random_integers(
                    low=0, high=(100 - 1), size=ys.shape[0])
                axarr[0].plot(np.array(X_test)[:, 0][:, None], np.array(
                    ys[range(ys.shape[0]), rnd_indices]), 'o', label="y", color=(0, 0.7, 0, 0.2))
                axarr[0].set_xlim([-8.5, 9.5])
                axarr[0].set_ylim([-5, 5])
                axarr[1].set_xlim([-8.5, 9.5])
                axarr[1].set_ylim([0, 25])
                plt.draw()
                plt.show()

            elif PLOT_KL:
                painter_kl.set_xdata(range((epoch + 1)))
                painter_kl.set_ydata(kl_div_means)
                plt.draw()

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))
            print(
                "  KL divergence:\t\t{:.6f} ({:.6f})".format(kl_mean, kl_stdn))
#             print(
#                 "  alien KL ({:.1f}):\t\t{:.6f} ; {:.6f}".format(alien_value, alien_kl, alien_kl / kl_mean))

        print("Done training.")


if __name__ == '__main__':
    pass