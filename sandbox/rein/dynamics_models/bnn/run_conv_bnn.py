import numpy as np
from sandbox.rein.dynamics_models.bnn.conv_bnn import ConvBNN
import lasagne
from sandbox.rein.dynamics_models.utils import sliding_mean, iterate_minibatches, plot_mnist_digit, load_dataset_MNIST
import time


def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):
    # Train convolutional BNN to autoencode MNIST digits.

    print('Training ...')

#     kl_div_means = []
#     kl_div_stdns = []
#     kl_all_values = []

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

        # Iterate over all minibatches and train on each of them.
        for batch in iterate_minibatches(X_train, T_train, model.batch_size, shuffle=True):

            # Fix old params for KL divergence computation.
            model.save_old_params()

            # Train current minibatch.
            inputs, _ = batch
            _train_err = model.train_fn(
                inputs, inputs.reshape(model.batch_size, 28 * 28))
#             pred = model.pred_fn(inputs)
#             plot_mnist_digit(pred[0].reshape(28, 28))

            train_err += _train_err
            train_batches += 1

#             # Calculate current minibatch KL.
#             kl_mb_closed_form = model.fn_kl()
#
#             kl_values.append(kl_mb_closed_form)
#             kl_all_values.append(kl_mb_closed_form)

#         # Calculate KL divergence variance over all minibatches.
#         kl_mean = np.mean(np.asarray(kl_values))
#         kl_stdn = np.std(np.asarray(kl_values))
#
#         kl_div_means.append(kl_mean)
#         kl_div_stdns.append(kl_stdn)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(
            train_err / train_batches))
#         print(
#             "  KL divergence:\t\t{:.6f} ({:.6f})".format(kl_mean, kl_stdn))

    pred = model.pred_fn(inputs)
    plot_mnist_digit(pred[0].reshape(28, 28))
    print("Done training.")


def main():

    num_epochs = 1000
    batch_size = 64

    print("Loading data ...")
    X_train, T_train, X_test, T_test = load_dataset_MNIST()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))

    print("Building model and compiling functions ...")
    bnn = ConvBNN(
        layers_disc=[
            dict(name='input', in_shape=(None, 1, 28, 28)),
            dict(name='convolution', n_filters=2, filter_size=(5, 5)),
            dict(name='pool', pool_size=(2, 2)),
            dict(name='upscale', scale_factor=2),
            dict(name='transconvolution', n_filters=2, filter_size=(5, 5)),
            dict(name='gaussian', n_units=32)
        ],
        n_out=28 * 28,
        n_batches=n_batches,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=10,
        prior_sd=0.5,
        update_likelihood_sd=True,
        learning_rate=0.0001
    )

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train,
          T_train=T_train, X_test=X_test, T_test=T_test)
    print('Done.')

if __name__ == '__main__':
    main()
