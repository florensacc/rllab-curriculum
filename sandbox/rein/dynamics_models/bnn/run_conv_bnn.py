import numpy as np
from sandbox.rein.dynamics_models.bnn.conv_bnn import ConvBNN
from utils import load_dataset_MNIST
import lasagne
from utils import sliding_mean, iterate_minibatches, plot_mnist_digit
import time

# Plotting params.
# ----------------
PLOT_WEIGHTS_INDIVIDUAL = False
PLOT_WEIGHTS_TOTAL = False
PLOT_OUTPUT = True
PLOT_OUTPUT_REGIONS = False
PLOT_KL = False

def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):
    # Train convolutional BNN to autoencode MNIST digits.

    training_data_start = 1000
    training_data_end = 1100

    print('Training ...')

    kl_div_means = []
    kl_div_stdns = []
    kl_all_values = []

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
            inputs, targets = batch
            _train_err = model.train_fn(inputs, targets)
            train_err += _train_err
            train_batches += 1

            # Calculate current minibatch KL.
            kl_mb_closed_form = model.f_kl_div_closed_form()

            kl_values.append(kl_mb_closed_form)
            kl_all_values.append(kl_mb_closed_form)

        # Calculate KL divergence variance over all minibatches.
        kl_mean = np.mean(np.asarray(kl_values))
        kl_stdn = np.std(np.asarray(kl_values))

        kl_div_means.append(kl_mean)
        kl_div_stdns.append(kl_stdn)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(
            train_err / train_batches))
        print(
            "  KL divergence:\t\t{:.6f} ({:.6f})".format(kl_mean, kl_stdn))

    print("Done training.")


def main():

    num_epochs = 1000
    batch_size = 1

    print("Loading data ...")
    X_train, T_train, X_test, T_test = load_dataset_MNIST()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))

    plot_mnist_digit(X_train[0].reshape(28,28))
    
    print("Building model and compiling functions ...")
    bnn = ConvBNN(
        n_in=4,
        n_hidden=[128],
        n_out=1,
        n_batches=n_batches,
        layers_type=[1, 1],
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=10,
        prior_sd=0.5,
        use_reverse_kl_reg=False,
        reverse_kl_reg_factor=1e-2
    )

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train,
              T_train=T_train, X_test=X_test, T_test=T_test)
    print('Done.')

if __name__ == '__main__':
    main()
