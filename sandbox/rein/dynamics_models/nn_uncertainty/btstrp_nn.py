from rllab.core.parameterized import Parameterized
from rllab.core.lasagne_powered import LasagnePowered
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
from sandbox.rein.dynamics_models.bnn.utils import iterate_minibatches, sliding_mean

PLOT_OUTPUT_REGIONS = True


class NN:

    def __init__(self,
                 n_in,
                 n_hidden,
                 n_out,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.softmax,
                 batch_size=100,
                 type='regression',
                 ):

        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.type = type
        self.n_batches = n_batches

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input)

    def loss_sym(self, input, target):
        loss = 0.5 * T.mean(T.square(target - self.pred_sym(input)))
        return loss

    def build_network(self):

        # Input layer
        network = lasagne.layers.InputLayer(shape=(self.batch_size, self.n_in))

        # Hidden layers
        for i in xrange(len(self.n_hidden)):
            # Probabilistic layer (1) or deterministic layer (0).
            network = lasagne.layers.DenseLayer(
                network, self.n_hidden[i], nonlinearity=self.transf)

        # Output layer
        network = lasagne.layers.DenseLayer(
            network, self.n_out, nonlinearity=self.outf)

        self.network = network

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        input_var = T.matrix('inputs', dtype=theano.config.floatX)
        target_var = T.matrix('targets', dtype=theano.config.floatX)

        # Loss function.
        loss = self.loss_sym(
            input_var, target_var)

        # Create update methods.
        params = lasagne.layers.get_all_params(self.network, trainable=True)
        updates = lasagne.updates.adam(
            loss, params, learning_rate=0.001)

        # Train/val fn.
        self.pred_fn = theano.function(
            [input_var], self.pred_sym(input_var), allow_input_downcast=True)
        self.train_fn = theano.function(
            [input_var, target_var], loss, updates=updates, allow_input_downcast=True)


class BootstrapNetwork(LasagnePowered, Parameterized):

    def __init__(
            self,
            n_in=4,
            n_hidden=[32],
            n_out=1,
            n_batches=None,
            trans_func=lasagne.nonlinearities.rectify,
            out_func=lasagne.nonlinearities.linear,
            batch_size=None,
            type='regression',
            n_networks=5
    ):
        self.batch_size = batch_size
        self.n_batches = n_batches

        self._networks = [
            NN(n_in=n_in,
               n_hidden=n_hidden,
               n_out=n_out,
               n_batches=n_batches,
               trans_func=trans_func,
               out_func=out_func,
               batch_size=batch_size,
               type='regression'
               )
            for _ in range(n_networks)
        ]

    def predict(self, x):
        ys = [n.pred_fn(x) for n in self._networks]
        return np.mean(ys, axis=0), np.std(ys, axis=0)

    def build_network(self):
        for n in self._networks:
            n.build_network()

    def build_model(self):
        for n in self._networks:
            n.build_model()

    def train_fn_ensemble(self, inputs, targets):
        rnd_index = np.random.random_integers(
                    low=0, high=(len(self._networks) - 1))
        train_err = self._networks[rnd_index].train_fn(inputs, targets)
        return train_err

    def train(self, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):

        training_data_start = 1000
        training_data_end = 1100

        print('Training ...')

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()

            # Iterate over all minibatches and train on each of them.
            for batch in iterate_minibatches(X_train, T_train, self.batch_size, shuffle=True):

                # Train current minibatch.
                inputs, targets = batch
                train_err += self.train_fn_ensemble(inputs, targets)
                train_batches += 1

            ##########################
            ### PLOT FUNCTIONALITY ###
            ##########################

            if PLOT_OUTPUT_REGIONS and epoch % 100 == 0 and epoch != 0:
                import matplotlib.pyplot as plt

                ys = []
                for network in self._networks:
                    y = [network.pred_fn(x[None, :])[0][0] for x in X_test]
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

                window_size = 10
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
                    low=0, high=(len(self._networks) - 1), size=ys.shape[0])
                axarr[0].plot(np.array(X_test)[:, 0][:, None], np.array(
                    ys[range(ys.shape[0]),rnd_indices]), 'o', label="y", color=(0, 0.7, 0, 0.2))
#                 axarr[0].xlim(xmin=-8.5, xmax=9.5)
#                 axarr[0].ylim(ymin=-5, ymax=5)
                axarr[0].set_xlim([-8.5, 9.5])
                axarr[0].set_ylim([-5, 5])
                axarr[1].set_xlim([-8.5, 9.5])
                axarr[1].set_ylim([0,25])
                plt.draw()
                plt.show()

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(
                train_err / train_batches))

        print("Done training.")


if __name__ == '__main__':
    pass
