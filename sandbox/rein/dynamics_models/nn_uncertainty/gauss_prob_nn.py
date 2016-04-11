from __future__ import print_function
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

############################
### Plotting global vars ###
############################
# Only works for regression

PLOT_WEIGHTS_INDIVIDUAL = False
PLOT_WEIGHTS_TOTAL = False
PLOT_OUTPUT = False

############################


def load_dataset_MNIST():
    """MNIST dataset loader"""

    import gzip

    def load_mnist_images(filename):
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 28 * 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            data = data.reshape(-1, 1)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    PATH = '/media/ssd/MNIST/'
    import os
    if not os.path.isdir(PATH):
        PATH = '/home/rein/PhD/MNIST/'

    X_train = load_mnist_images(PATH + 'train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(PATH + 'train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(PATH + 't10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(PATH + 't10k-labels-idx1-ubyte.gz')

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_test, y_test


# def load_dataset_1Dregression():
#     """Synthetic 1D regression data loader"""
# 
#     def generate_synthetic_data(dat_size=1.e4):
#         rng = np.random.RandomState(1234)
#         x = rng.uniform(0., 1., dat_size).reshape((dat_size, 1))
#         v = rng.normal(0, 0.02, size=x.shape)
#         # TODO: Write y.
#         y = x + 0.3 * np.sin(2. * np.pi * (x + v)) + 0.3 * \
#             np.sin(4. * np.pi * (x + v)) + v
#         y += np.random.randint(low=0, high=2, size=(len(y), 1)) * 0.5
#         # 90% for training, 10% testing
#         train_x = x[:len(x) * 0.9]
#         train_y = y[:len(y) * 0.9]
#         test_x = x[len(x) * 0.9:]
#         test_y = y[len(y) * 0.9:]
# 
#         x2 = rng.uniform(1., 10., dat_size / 10).reshape((dat_size / 10, 1))
#         y2 = x2 * 0. + 0.5
#         test_x = np.vstack([test_x, x2])
#         test_y = np.vstack([test_y, y2])
#         x2 = rng.uniform(-9., 0., dat_size / 10).reshape((dat_size / 10, 1))
#         y2 = x2 * 0. + 0.5
#         test_x = np.vstack([x2, test_x])
#         test_y = np.vstack([y2, test_y])
#         return (train_x, train_y), (test_x, test_y)
# 
#     return generate_synthetic_data(1e4)

def load_dataset_1Dregression():
    """Synthetic 1D regression data loader"""

    def generate_synthetic_data(dat_size=1.e4):
        rng = np.random.RandomState(1234)
        x = rng.uniform(0., 1., dat_size).reshape((dat_size, 1))
        v = rng.normal(0, 0.02, size=x.shape)
        # TODO: Write y.
        y = x + 0.3 * np.sin(2. * np.pi * (x + v)) + 0.3 * \
            np.sin(4. * np.pi * (x + v)) + v
        y += np.random.randint(low=-1, high=2, size=(len(y), 1)) * 2

        # 90% for training, 10% testing
        train_x = x[:len(x) * 0.9]
        train_y = y[:len(y) * 0.9]
        test_x = x[len(x) * 0.9:]
        test_y = y[len(y) * 0.9:]

        x2 = rng.uniform(1., 10., dat_size / 10).reshape((dat_size / 10, 1))
        y2 = x2 * 0. + 0.5
        test_x = np.vstack([test_x, x2])
        test_y = np.vstack([test_y, y2])
        x2 = rng.uniform(-9., 0., dat_size / 10).reshape((dat_size / 10, 1))
        y2 = x2 * 0. + 0.5
        test_x = np.vstack([x2, test_x])
        test_y = np.vstack([y2, test_y])

        train_x = np.hstack([train_x, train_x**2, train_x**3])
        test_x = np.hstack([test_x, test_x**2, test_x**3])
        return (train_x, train_y), (test_x, test_y)

    return generate_synthetic_data(1e4)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

class GenLayer(lasagne.layers.Layer):
    """Generating Gaussian layer: takes as input mean and std params, outputs sample from Gaussian distribution.
    For this to work we need to separate the p(y|h) part and the p(h|x) part, in two subsampling steps because
    now the backpropagation wont flow through the h sampling process...

    This layer has no params.
    """

    def __init__(self,
                 incoming,
                 num_units,
                 batch_size=-1,
                 prior_sd=None,
                 **kwargs):
        super(GenLayer, self).__init__(incoming, **kwargs)

        self._srng = RandomStreams()

        # Set vars.
        self.num_units = num_units
        self.num_inputs = int(np.prod(self.input_shape[1:]))
        self.prior_sd = prior_sd
        self.batch_size = batch_size

    def get_sample(self, p):
        sample = self._srng.binomial((self.batch_size, self.num_units), p=p)
        return sample

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = lasagne.nonlinearities.sigmoid(input)
        sample = self.get_sample(p=activation)
        out = sample * input
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

class ProbLayer(lasagne.layers.Layer):
    """Probabilistic layer: for now uses Gaussian weights (2 params)."""

    def __init__(self,
                 incoming,
                 num_units,
                 mu=lasagne.init.Normal(std=1., mean=0.),
                 rho=lasagne.init.Normal(std=1., mean=1.),
                 b_mu=lasagne.init.Normal(std=0.35, mean=0.),
                 b_rho=lasagne.init.Normal(std=0.35, mean=0.),
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

        # Old param backup for kl div calc.
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
        epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=0.05,
                                    dtype=theano.config.floatX)
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + T.log(1 + T.exp(self.rho)) * epsilon
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        # (paper step 1)
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=0.05,
                                    dtype=theano.config.floatX)
        b = self.b_mu + T.log(1 + T.exp(self.b_rho)) * epsilon
        return b

    def save_old_params(self):
        self.mu_old.set_value(self.mu.get_value())
        self.rho_old.set_value(self.rho.get_value())
        self.b_mu_old.set_value(self.b_mu.get_value())
        self.b_rho_old.set_value(self.b_rho.get_value())

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.get_W()) + \
            self.get_b().dimshuffle('x', 0)

        return self.nonlinearity(activation)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class ProbNN:

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
                 reg=1.,
                 type='regression'):

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
        self.stdn = 0.5
        self.reg = reg

        self._srng = RandomStreams()

    def _log_prob_normal(self, input, mu=0., sigma=0.01):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)
    
    def _log_prob_normal_sym(self, input, mu=0., sigma=0.01):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return log_normal

    def pred_sym(self, input):
        # Mean is a matrix of batch_size rows.
        mean = self.pred_mean(input)
        stdn = self.pred_stdn(input)
        epsilon = self._srng.normal(size=(self.batch_size, 1), avg=0., std=1.)
        out = mean + epsilon * stdn
        return out

    def pred_mean(self, input):
        return lasagne.layers.get_output(self.network_mean, input)

    def pred_stdn(self, input):
        return T.log(1 + T.exp(lasagne.layers.get_output(self.network_stdn, input)))

    def get_log_p_D_given_w(self, input, target):
        log_p_D_given_w,sum_lqw = 0.,0.
        # MC samples.
        for _ in xrange(self.n_samples):
            if self.type == 'regression':
                lqw = self._log_prob_normal_sym(
                    target, self.pred_mean(input), self.pred_stdn(input))
                exp_lqw = T.clip(T.exp(lqw), 1e-20, 1e20)
                log_p_D_given_w += lqw * exp_lqw
                sum_lqw += exp_lqw
            elif self.type == 'classification':
                prediction = self.pred_sym(input)
                log_p_D_given_w += T.sum(
                    T.log(prediction)[T.arange(target.shape[0]), T.cast(target[:, 0], dtype='int64')])
                
                
        # Calculate importance sampling ratio:
        mean_lqw = sum_lqw / self.n_samples
        log_p_D_given_w /= mean_lqw
        log_p_D_given_w = T.sum(log_p_D_given_w)
        
        return log_p_D_given_w

    def get_q_w(self):
        log_q_w = 0.

        # MC samples.
        for _ in xrange(self.n_samples):
            # Calculate variational posterior log(q(w)) and prior log(p(w)).
            layers_mean = lasagne.layers.get_all_layers(self.network_mean)[1:]
            layers_stdn = lasagne.layers.get_all_layers(self.network_stdn)[1:]
            layers = layers_mean + layers_stdn
            for layer in layers:
                if layer.name == 'problayer':
                    W = layer.get_W()
                    b = layer.get_b()
                    log_q_w += self._log_prob_normal(W,
                                                     layer.mu, T.log(1 + T.exp(layer.rho)))
                    log_q_w += self._log_prob_normal(b,
                                                     layer.b_mu, T.log(1 + T.exp(layer.b_rho)))
        return log_q_w

    def get_p_w(self):
        log_p_w = 0.

        # MC samples.
        for _ in xrange(self.n_samples):
            layers_mean = lasagne.layers.get_all_layers(self.network_mean)[1:]
            layers_stdn = lasagne.layers.get_all_layers(self.network_stdn)[1:]
            layers = layers_mean + layers_stdn
            for layer in layers:
                if layer.name == 'problayer':
                    W = layer.get_W()
                    b = layer.get_b()
                    log_p_w += self._log_prob_normal(W, 0., self.prior_sd)
                    log_p_w += self._log_prob_normal(b, 0., self.prior_sd)
        return log_p_w

    def get_kl_div(self):
        kl_div = 0.
        # MC samples.
        for _ in xrange(self.n_samples):
            # Calculate variational posterior log(q(w)) and prior log(p(w)).
            layers_mean = lasagne.layers.get_all_layers(self.network_mean)[1:]
            layers_stdn = lasagne.layers.get_all_layers(self.network_stdn)[1:]
            layers = layers_mean + layers_stdn
            for layer in layers:
                if layer.name == 'problayer':
                    W = layer.get_W()
                    b = layer.get_b()
                    kl_div += self._log_prob_normal(W,
                                                    layer.mu, T.log(1 + T.exp(layer.rho)))
                    kl_div += self._log_prob_normal(b,
                                                    layer.b_mu, T.log(1 + T.exp(layer.b_rho)))
                    kl_div -= self._log_prob_normal(W,
                                                    layer.mu_old, T.log(1 + T.exp(layer.rho_old)))
                    kl_div -= self._log_prob_normal(b,
                                                    layer.b_mu_old, T.log(1 + T.exp(layer.b_rho_old)))
        kl_div /= self.n_samples
        return kl_div

    def get_loss_sym(self, input, target):

        log_p_D_given_w = self.get_log_p_D_given_w(input, target)
        log_q_w = self.get_q_w()
        log_p_w = self.reg * self.get_p_w()

        # Calculate loss function.
        loss = ((log_q_w - log_p_w) / self.n_batches -
                log_p_D_given_w) / self.batch_size
        loss /= self.n_samples

        return loss

    def save_old_params(self):
        layers_mean = lasagne.layers.get_all_layers(self.network_mean)[1:]
        layers_stdn = lasagne.layers.get_all_layers(self.network_stdn)[1:]
        layers = layers_mean + layers_stdn
        for layer in layers:
            if layer.name == 'problayer':
                layer.save_old_params()

    def build_network(self):

        # Input layer
        input = lasagne.layers.InputLayer(shape=(self.batch_size, self.n_in))


        ### Mean network ###
        network = input
        # Hidden layers
        for i in xrange(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            if self.layers_type[i] == 1:
                network = ProbLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, prior_sd=self.prior_sd, name='problayer')
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, W=lasagne.init.Normal(std=1, mean=0.))
            if i == 0:
                network = GenLayer(
                    network, self.n_hidden[i], prior_sd=self.prior_sd, batch_size=self.batch_size, name='genlayer')

        # Output layer
        if self.layers_type[len(self.n_hidden)] == 1:
            network = ProbLayer(
                network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name='problayer')
        else:
            network = lasagne.layers.DenseLayer(
                network, self.n_out, nonlinearity=self.outf, W=lasagne.init.Normal(std=1, mean=0.))

        self.network_mean = network

        ### Stdn network ###
        network = input
        # Output layer
        if self.layers_type[len(self.n_hidden)] == 1:
            network = ProbLayer(
                network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name='problayer')
        else:
            network = lasagne.layers.DenseLayer(
                network, self.n_out, nonlinearity=self.outf)

        self.network_stdn = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs')
        target_var = T.matrix('targets')

        # Loss function
        loss = self.get_loss_sym(
            input_var, target_var)

        # Create update methods.
        params_mean = lasagne.layers.get_all_params(
            self.network_mean, trainable=True)
        params_stnd = lasagne.layers.get_all_params(
            self.network_stdn, trainable=True)
        params = params_mean + params_stnd
    #     updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.001, momentum=0.9)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=0.01)

        # Test functions
        test_prediction = self.pred_sym(input_var)
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var[:, 0]),
                          dtype=theano.config.floatX)

        # Train/val functions
        self.pred_fn = theano.function([input_var], self.pred_sym(input_var))
        self.train_fn = theano.function(
            [input_var, target_var], loss, updates=updates)
        self.val_fn = theano.function(
            [input_var, target_var], [loss, test_acc])

        # Diagnostic functions
#         self.f_q_w = theano.function([], self.get_q_w())
#         self.f_p_w = theano.function([], self.get_p_w())
#         self.f_kl_div = theano.function([], self.get_kl_div())

    def train(self, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):

        #         import matplotlib.pyplot as plt
        #         y = []
        #         for i in xrange(1000):
        #             _y = self.pred_fn(X_test[0][:, None])[0][0]
        #             y.append(_y)
        #         plt.hist(y, 100)
        #         plt.show()

        # Print weights from this layer.
        layer = lasagne.layers.get_all_layers(self.network_mean)[-1]

        if PLOT_WEIGHTS_TOTAL:
            import matplotlib.pyplot as plt
            plt.ion()
            sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
            mean = layer.mu.eval().ravel()
            painter, = plt.plot(mean, sd, 'o', color=(1.0, 0, 0, 0.2))
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
            painter = []
            for i in xrange(n_plots_v):
                for j in xrange(n_plots_h):
                    hl, = axarr[i][j].plot(x, x)
                    axarr[i][j].set_ylim(ymin=0, ymax=2)
                    painter.append(hl)
            plt.draw()
            plt.show()

        elif PLOT_OUTPUT:
            import matplotlib.pyplot as plt
            plt.ion()
#             y = [self.pred_fn(x[:, None])[0][0] for x in X_test]
            y = []
            for batch in iterate_minibatches(X_test, T_test, self.batch_size, shuffle=False):
                inputs, targets = batch
                y.append(self.pred_fn(inputs))
            y = list(np.array(y).flatten())
            _ = plt.figure()
            plt.plot(np.array(X_test[1000:2000])[:,0], np.array(
                T_test[1000:2000]), 'o', label="t", color=(1.0, 0, 0, 0.1))
            painter, = plt.plot(np.array(X_test[1000:2000])[:,0], np.array(
                y[1000:2000]), 'o', label="y", color=(0, 0.7, 0, 0.2))
            plt.xlim(xmin=-10, xmax=11)
            plt.ylim(ymin=-5, ymax=5)
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

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, T_train, self.batch_size, shuffle=True):
                inputs, targets = batch
                # Save old params for every update.
                self.save_old_params()
                # Update model weights based on current minibatch.
                train_err += self.train_fn(inputs, targets)
                # Calculate kl divergence between q(w') and q(w).
                train_batches += 1

#             print('kl last update %s' % self.f_kl_div())
#             print('log_q_w %s' %
#                   (self.f_q_w() / self.n_batches / self.batch_size))
#             print('log_p_w %s' %
#                   (self.f_p_w() / self.n_batches / self.batch_size))

            if PLOT_WEIGHTS_TOTAL:
                if epoch % 1 == 0:
                    sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
                    mean = layer.mu.eval().ravel()
                    painter.set_xdata(mean)
                    painter.set_ydata(sd)
                    plt.draw()

            elif PLOT_WEIGHTS_INDIVIDUAL:
                if epoch % 1 == 0:
                    for i in xrange(n_plots):
                        w_mu = layer.mu.eval()[i, 0]
                        w_rho = layer.rho.eval()[i, 0]
                        w_sigma = np.log(1 + np.exp(w_rho))
                        y = normal(x, w_mu, w_sigma)
                        painter[i].set_ydata(y)
                    plt.draw()

            elif PLOT_OUTPUT:
                if epoch % 1 == 0:
                    #                     y = [self.pred_fn(x[:, None])[0][0] for x in X_test]
                    y = []
                    for batch in iterate_minibatches(X_test, T_test, self.batch_size, shuffle=False):
                        inputs, targets = batch
                        y.append(self.pred_fn(inputs))
                    y = list(np.array(y).flatten())
                    painter.set_ydata(y[1000:2000])
                    plt.draw()

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))

            # After training, we compute and print the test error:
            test_err = 0
            test_acc = 0
            test_batches = 0
            for batch in iterate_minibatches(X_test, T_test, self.batch_size, shuffle=False):
                inputs, targets = batch
                err, acc = self.val_fn(inputs, targets)
                test_err += err
                test_acc += acc
                test_batches += 1
            print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
            print("  test accuracy:\t\t{:.2f} %".format(
                test_acc / test_batches * 100))

        print('Done training.')

        import matplotlib.pyplot as plt
        plt.ioff()
        y = [self.pred_fn(x[:, None])[0][0] for x in X_test]
        _ = plt.figure()
        plt.plot(np.array(X_test[1000:2000]), np.array(
            T_test[1000:2000]), 'o', label="t", color=(1.0, 0, 0, 0.1))
        painter, = plt.plot(np.array(X_test), np.array(
            y), 'o', label="y", color=(0, 0.7, 0, 0.2))
        plt.xlim(xmin=-10, xmax=10)
        plt.ylim(ymin=-5, ymax=5)
        plt.draw()
        # For Jupyter notebook
        try:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        except:
            pass
        plt.show()

        # Optionally, you could now dump the network weights to a file like
        # this:
#         np.savez(
#             'model.npz', *lasagne.layers.get_all_param_values(self.network))

        # And load them again later on like this:
        # with np.load('model.npz') as f:
        #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    pass
