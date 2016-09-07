
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
        self.rho = 0.5
        self.batch_size = batch_size
#         self.W = self.add_param(
# lasagne.init.Normal(std=1., mean=0.), (self.num_inputs, self.num_units),
# name='gen_W')
        self.mask = np.zeros((self.batch_size, self.num_units))
        #self.mask[:,:] = 1.

    def get_sample(self, p):
        #         epsilon = self._srng.uniform((self.batch_size, self.num_units), 0, 1)
        #         sample = lasagne.nonlinearities.sigmoid((epsilon - p) * 100)

        sample = self._srng.binomial((self.batch_size, self.num_units), p=p)
        return sample

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = lasagne.nonlinearities.sigmoid(input)
        sample = self.get_sample(p=activation)
        out = activation * self.mask + sample * (1 - self.mask)
        out = out * input
        return out

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)


class VBNNLayer(lasagne.layers.Layer):
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
        super(VBNNLayer, self).__init__(incoming, **kwargs)

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

    def get_W(self):
        # Here we generate random epsilon values from a normal distribution
        # (paper step 1)
        epsilon = self._srng.normal(size=(self.num_inputs, self.num_units), avg=0., std=self.prior_sd,
                                    dtype=theano.config.floatX)
        # Here we calculate weights based on shifting and rescaling according
        # to mean and variance (paper step 2)
        W = self.mu + T.log(1 + T.exp(self.rho)) * epsilon
        return W

    def get_b(self):
        # Here we generate random epsilon values from a normal distribution
        # (paper step 1)
        epsilon = self._srng.normal(size=(self.num_units,), avg=0., std=self.prior_sd,
                                    dtype=theano.config.floatX)
        b = self.b_mu + T.log(1 + T.exp(self.b_rho)) * epsilon
        return b

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


class VBNN:

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

    def _log_prob_normalx(self, input, mu=0., sigma=0.01):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)

    def _log_prob_normal(self, input, mu=0., sigma=0.01):
        log_normal = - \
            np.log(sigma) - np.log(np.sqrt(2 * np.pi)) - \
            np.square(input - mu) / (2 * np.square(sigma))
        return log_normal

    def _log_prob_normal_sym(self, input, mu=0., sigma=0.01):
        log_normal = - \
            T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - \
            T.square(input - mu) / (2 * T.square(sigma))
        return log_normal

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def get_loss_sym(self, input, target):

        log_q_w, log_p_w, log_p_D_given_w, sum_lqw, sum_ratio = 0., 0., 0., 0., 0.

        # MC samples.
        for _ in range(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self.type == 'regression':
                lqw = self._log_prob_normal_sym(
                    target, prediction, self.prior_sd)
                exp_lqw = T.clip(T.exp(lqw), 1e-20, 1e20)
                log_p_D_given_w += lqw * exp_lqw
                sum_lqw += exp_lqw
            elif self.type == 'classification':
                log_p_D_given_w += T.sum(
                    T.log(prediction)[T.arange(target.shape[0]), T.cast(target[:, 0], dtype='int64')])
            # Calculate variational posterior log(q(w)) and prior log(p(w)).
            layers = lasagne.layers.get_all_layers(self.network)[1:]
            for layer in layers:
                if layer.name == 'problayer':
                    W = layer.get_W()
                    b = layer.get_b()
                    log_q_w += T.sum(self._log_prob_normal_sym(W,
                                                               layer.mu, T.log(1 + T.exp(layer.rho))))
                    log_q_w += T.sum(self._log_prob_normal_sym(b,
                                                               layer.b_mu, T.log(1 + T.exp(layer.b_rho))))
                    log_p_w += T.sum(self._log_prob_normal_sym(W,
                                                               0., self.prior_sd))
                    log_p_w += T.sum(self._log_prob_normal_sym(b,
                                                               0., self.prior_sd))

        # Calculate importance sampling ratio:
        mean_lqw = sum_lqw / self.n_samples
        log_p_D_given_w /= mean_lqw
        log_p_D_given_w = T.sum(log_p_D_given_w)

        # Calculate loss function.
        loss = ((log_q_w - log_p_w) / self.n_batches -
                log_p_D_given_w) / self.batch_size
        loss /= self.n_samples

        return loss

    def build_network(self):

        # Input layer
        network = lasagne.layers.InputLayer(shape=(self.batch_size, self.n_in))

        # Hidden layers
        for i in range(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            if self.layers_type[i] == 1:
                network = VBNNLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, prior_sd=self.prior_sd, name='problayer')
            else:
                network = lasagne.layers.DenseLayer(
                    network, self.n_hidden[i], nonlinearity=self.transf, W=lasagne.init.Normal(std=1, mean=0.))
            if i == 0:
                network = GenLayer(
                    network, self.n_hidden[i], prior_sd=self.prior_sd, batch_size=self.batch_size, name='genlayer')

        # Output layer
        if self.layers_type[len(self.n_hidden)] == 1:
            network = VBNNLayer(
                network, self.n_out, nonlinearity=self.outf, prior_sd=self.prior_sd, name='problayer')
        else:
            network = lasagne.layers.DenseLayer(
                network, self.n_out, nonlinearity=self.outf, W=lasagne.init.Normal(std=1, mean=0.))

        self.network = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs')
        target_var = T.matrix('targets')

        # Loss function.
        loss = self.get_loss_sym(
            input_var, target_var)

        # Create update methods.
        params = lasagne.layers.get_all_params(self.network)
        print(params)
    #     updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.001, momentum=0.9)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=0.001)

        # Test acc.
        test_prediction = self.pred_sym(input_var)
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var[:, 0]),
                          dtype=theano.config.floatX)

        # Train/val fn.
        self.pred_fn = theano.function([input_var], self.pred_sym(input_var))
        self.train_fn = theano.function(
            [input_var, target_var], loss, updates=updates)
        self.val_fn = theano.function(
            [input_var, target_var], [loss, test_acc])

    def train(self, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):

        #         import matplotlib.pyplot as plt
        #         plt.ion()
        #         y = []
        #         for i in xrange(100):
        #             _y = self.pred_fn(X_test[0:self.batch_size])[0][0]
        #             y.append(_y)
        #         plt.hist(y, self.batch_size)
        #         plt.show()

        def get_loss_sym(input, target):

            log_q_w, log_p_w, log_p_D_given_w, sum_lqw = 0., 0., 0., 0.

            # MC samples.
            for _ in range(self.n_samples):
                # Make prediction.
                prediction = self.pred_fn(input)
                # Calculate model likelihood log(P(D|w)).
                if self.type == 'regression':
                    lqw = self._log_prob_normal(
                        target, prediction, self.prior_sd)
                    log_p_D_given_w += lqw * np.exp(lqw) - np.exp(lqw)
                    sum_lqw += np.exp(lqw)
                elif self.type == 'classification':
                    log_p_D_given_w += np.sum(
                        np.log(prediction)[np.arange(target.shape[0]), np.cast(target[:, 0], dtype='int64')])
                # Calculate variational posterior log(q(w)) and prior
                # log(p(w)).
                layers = lasagne.layers.get_all_layers(self.network)[1:]
                for layer in layers:
                    if layer.name == 'problayer':
                        W = layer.get_W()
                        b = layer.get_b()
                        log_q_w += self._log_prob_normal(W,
                                                         layer.mu, np.log(1 + np.exp(layer.rho)))
                        log_q_w += self._log_prob_normal(b,
                                                         layer.b_mu, np.log(1 + np.exp(layer.b_rho)))
                        log_p_w += self._log_prob_normal(W, 0., self.prior_sd)
                        log_p_w += self._log_prob_normal(b, 0., self.prior_sd)

            # Calculate importance sampling ratio:
            mean_lqw = sum_lqw / self.n_samples
            log_p_D_given_w /= np.clip(mean_lqw, a_min=0, a_max=np.exp(10))
            log_p_D_given_w = np.sum(log_p_D_given_w)

            # Calculate loss function.
            loss = ((log_q_w - log_p_w) / self.n_batches -
                    log_p_D_given_w) / self.batch_size
            loss /= self.n_samples

            return loss

        # Print weights from this layer.
        layer = lasagne.layers.get_all_layers(self.network)[-1]

        print(lasagne.layers.get_all_layers(self.network))
        show_layer = lasagne.layers.get_all_layers(self.network)[1]

        y = []
        for batch in iterate_minibatches(X_test, T_test, self.batch_size, shuffle=False):
            inputs, targets = batch
            y.append(lasagne.layers.get_output(show_layer, inputs).eval())
        print(y)

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
            for i in range(n_plots_v):
                for j in range(n_plots_h):
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

            #             if epoch % 10 == 0:
            #                 import matplotlib.pyplot as plt
            #                 y = []
            #                 for i in xrange(1000):
            #                     _y = self.pred_fn(X_test[0:self.batch_size])[0][0]
            #                     y.append(_y)
            #                 plt.clf()
            #                 plt.hist(y, self.batch_size)
            #                 plt.draw()

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, T_train, self.batch_size, shuffle=True):
                inputs, targets = batch
#                 print(get_loss_sym(inputs, targets))
                train_err += self.train_fn(inputs, targets)
                train_batches += 1

            if PLOT_WEIGHTS_TOTAL:
                if epoch % 1 == 0:
                    sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
                    mean = layer.mu.eval().ravel()
                    painter.set_xdata(mean)
                    painter.set_ydata(sd)
                    plt.draw()

            elif PLOT_WEIGHTS_INDIVIDUAL:
                if epoch % 1 == 0:
                    for i in range(n_plots):
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
        if PLOT_OUTPUT:
            plt.ioff()
            #                     y = [self.pred_fn(x[:, None])[0][0] for x in X_test]
            y = []
            for batch in iterate_minibatches(X_test, T_test, self.batch_size, shuffle=False):
                inputs, targets = batch
                y.append(self.pred_fn(inputs))
            y = list(np.array(y).flatten())
            painter.set_ydata(y[1000:2000])
            plt.draw()
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
