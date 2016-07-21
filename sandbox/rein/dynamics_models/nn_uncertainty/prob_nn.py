from __future__ import print_function
import numpy as np
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable

# ----------------
USE_REPARAMETRIZATION_TRICK = True
# ----------------


def softplus(rho):
    """Transformation for allowing rho in \mathbb{R}, rather than \mathbb{R}_+

    This makes sure that we don't get negative stds. However, a downside might be
    that we have little gradient on close to 0 std (= -inf using this transformation).
    """
    return T.log(1 + T.exp(rho))


def inv_softplus(sigma):
    """Reverse softplus transformation."""
    return np.log(np.exp(sigma) - 1)


class GaussianLayer(lasagne.layers.Layer):
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
        super(GaussianLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        # Set vars.
        self.nonlinearity = nonlinearity
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.num_units = num_units
        self.prior_sd = prior_sd

        prior_rho = inv_softplus(self.prior_sd)

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
        self.b_rho_old = self.add_param(
            np.ones((self.num_units,)),
            (self.num_units,),
            name="b_rho_old",
            regularizable=False,
            trainable=False,
            oldparam=True
        )

    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + softplus(self.rho) * epsilon
        self.W = W
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable
        b = self.b_mu + softplus(self.b_rho) * epsilon
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
        delta = T.dot(T.square(input), T.square(softplus(
            self.rho))) + T.square(softplus(self.b_rho)).dimshuffle('x', 0)
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=1.,
                                    dtype=theano.config.floatX)  # @UndefinedVariable

        activation = gamma + T.sqrt(delta) * epsilon

        return self.nonlinearity(activation)

    def save_params(self):
        """Save old parameter values for KL calculation."""
        self.mu_old.set_value(self.mu.get_value())
        self.rho_old.set_value(self.rho.get_value())
        self.b_mu_old.set_value(self.b_mu.get_value())
        self.b_rho_old.set_value(self.b_rho.get_value())

    def load_prev_params(self):
        """Reset to old parameter values for KL calculation."""
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
        kl_div = self.kl_div_p_q(
            self.mu, softplus(self.rho), self.mu_old, softplus(self.rho_old))
        kl_div += self.kl_div_p_q(self.b_mu, softplus(self.b_rho),
                                  self.b_mu_old, softplus(self.b_rho_old))
        return kl_div

    def kl_div_old_new(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, softplus(self.rho_old), self.mu, softplus(self.rho))
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  softplus(self.b_rho_old), self.b_mu, softplus(self.b_rho))
        return kl_div

    def kl_div_new_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu, softplus(self.rho), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu,
                                  softplus(self.b_rho), 0., self.prior_sd)
        return kl_div

    def kl_div_old_prior(self):
        kl_div = self.kl_div_p_q(
            self.mu_old, softplus(self.rho_old), 0., self.prior_sd)
        kl_div += self.kl_div_p_q(self.b_mu_old,
                                  softplus(self.b_rho_old), 0., self.prior_sd)
        return kl_div

    def kl_div_prior_new(self):
        kl_div = self.kl_div_p_q(
            0., self.prior_sd, self.mu,  softplus(self.rho))
        kl_div += self.kl_div_p_q(0., self.prior_sd,
                                  self.b_mu, softplus(self.b_rho))
        return kl_div

    def get_output_for(self, input, **kwargs):
        if USE_REPARAMETRIZATION_TRICK:
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


class ProbNN(LasagnePowered, Serializable):
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
                 prior_sd=0.5,
                 use_reverse_kl_reg=False,
                 reverse_kl_reg_factor=0.1,
                 likelihood_sd=0.05,
                 second_order_update=False,
                 learning_rate=0.0001,
                 compression=False,
                 information_gain=True,
                 ):

        Serializable.quick_init(self, locals())
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
        self.n_batches = n_batches
        self.use_reverse_kl_reg = use_reverse_kl_reg
        self.reverse_kl_reg_factor = reverse_kl_reg_factor
        self.likelihood_sd = likelihood_sd
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.compression = compression
        self.information_gain = information_gain
        
        assert self.information_gain or self.compression

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()
        
    def save_params(self):
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.save_params()

    def load_prev_params(self):
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.load_prev_params()

    def compr_impr_kl(self):
        """KL divergence KL[old_param||new_param]"""
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def surprise(self):
        surpr = 0.
        if self.compression:
            surpr += self.compr_impr_kl()
        if self.information_gain:
            surpr += self.inf_gain()
        return surpr

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = filter(lambda l: isinstance(l, GaussianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_prior_new() for l in layers)

    def _log_prob_normal(self, input, mu=0., sigma=1.):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def loss(self, input, target):
 
        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            _log_p_D_given_w.append(self._log_prob_normal(
                target, prediction, self.likelihood_sd))
        log_p_D_given_w = sum(_log_p_D_given_w)
        
        # Calculate variational posterior log(q(w)) and prior log(p(w)).
        kl = self.log_p_w_q_w_kl()
        if self.use_reverse_kl_reg:
            kl += self.reverse_kl_reg_factor * \
                self.reverse_log_p_w_q_w_kl()
 
        # Calculate loss function.
        return kl / self.n_batches - log_p_D_given_w / self.n_samples

    
#     def loss(self, input, target):
# 
#         # MC samples.
#         _log_p_D_given_w = []
#         for _ in xrange(self.n_samples):
#             # Make prediction.
#             prediction = self.pred_sym(input)
#             # Calculate model likelihood log(P(D|w)).
#             _log_p_D_given_w.append(self._log_prob_normal(
#                 target, prediction, self.likelihood_sd))
#         log_p_D_given_w = sum(_log_p_D_given_w)
#         # Calculate variational posterior log(q(w)) and prior log(p(w)).
#         kl = self.log_p_w_q_w_kl()
#         if self.use_reverse_kl_reg:
#             kl += self.reverse_kl_reg_factor * \
#                 self.reverse_log_p_w_q_w_kl()
# 
#         # Calculate loss function.
#         return kl / self.n_batches - log_p_D_given_w / self.n_samples

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
            log_p_D_given_w += self._log_prob_normal(
                    target, prediction, self.prior_sd)

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

    def get_loss_sym(self, input, target):
        raise Exception("Deprecated")

        log_q_w, log_p_w, log_p_D_given_w = 0., 0., 0.

        # MC samples.
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.type == 'regression':
                log_p_D_given_w += self._log_prob_normal(
                        target, prediction, self.prior_sd)

            # Calculate variational posterior log(q(w)) and prior log(p(w)).
            layers = lasagne.layers.get_all_layers(self.network)[1:]
            for layer in layers:
                if isinstance(layer, GaussianLayer):
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
        network = lasagne.layers.InputLayer(shape=(1, self.n_in))

        # Hidden layers
        for i in xrange(len(self.n_hidden)):
            if self.layers_type[i] == 'gaussian':
                network = GaussianLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, prior_sd=self.prior_sd)
            elif self.layers_type[i] == 'deterministic':
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf)

        # Output layer
        if self.layers_type[len(self.n_hidden)] == 'gaussian':
            network = GaussianLayer(
                network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd)
        elif self.layers_type[len(self.n_hidden)] == 'deterministic':
            network = lasagne.layers.DenseLayer(
                network, self.n_out, nonlinearity=self.outf)

        self.network = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)  # @UndefinedVariable
        target_var = T.matrix('targets',
                              dtype=theano.config.floatX)  # @UndefinedVariable

        # Loss function.
        loss = self.loss(
            input_var, target_var)
        loss_only_last_sample = self.get_loss_only_last_sample(
            input_var, target_var)

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
        self.f_kl_div_closed_form = theano.function(
                [], self.surprise(), allow_input_downcast=True)

if __name__ == '__main__':
    pass
