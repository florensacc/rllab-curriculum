import numpy as np
from sandbox.rein.dynamics_models.bnn.conv_bnn import ConvBNN
import lasagne
from sandbox.rein.dynamics_models.utils import iterate_minibatches, plot_mnist_digit, load_dataset_MNIST
import time


def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None, plt=None, ax=None):
    # Train convolutional BNN to autoencode MNIST digits.

    print('Training ...')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

        pred = model.pred_fn(X_train[0][:, None])
        plot_mnist_digit(pred[0].reshape(28, 28), plt, ax)

        # Iterate over all minibatches and train on each of them.
        for batch in iterate_minibatches(X_train, T_train, model.batch_size, shuffle=True):

            # Fix old params for KL divergence computation.
            model.save_old_params()

            # Train current minibatch.
            inputs, _ = batch
            _train_err = model.train_fn(
                inputs, inputs.reshape(model.batch_size, 28 * 28), 1.0)

            train_err += _train_err
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(
            train_err / train_batches))

    pred = model.pred_fn(inputs)
    plot_mnist_digit(pred[0].reshape(28, 28), plt, ax)
    print("Done training.")


def main():

    num_epochs = 1000
    batch_size = 64

    print("Loading data ...")
    X_train, T_train, X_test, T_test = load_dataset_MNIST()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))

    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plot_mnist_digit(X_train[0][0], plt, ax)

    print("Building model and compiling functions ...")
    bnn = ConvBNN(
        layers_disc=[
            dict(name='input', in_shape=(None, 1, 28, 28)),
            dict(name='convolution', n_filters=2, filter_size=(5, 5)),
            dict(name='pool', pool_size=(2, 2)),
            dict(name='upscale', scale_factor=2),
            dict(name='transconvolution', n_filters=2, filter_size=(5, 5))
            #             dict(name='gaussian', n_units=8),
        ],
        n_out=28 * 28,
        n_batches=n_batches,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=2,
        prior_sd=0.5,
        update_likelihood_sd=True,
        learning_rate=0.001,
        group_variance_by=ConvBNN.GroupVarianceBy.UNIT,
        use_local_reparametrization_trick=False,
        likelihood_sd_init=1.0,
        output_type=ConvBNN.OutputType.REGRESSION,
        surprise_type=ConvBNN.SurpriseType.BALD,
        num_classes=None,
        num_output_dim=None,
        disable_variance=False,
        second_order_update=True,
        debug=True
    )

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train,
          T_train=T_train, X_test=X_test, T_test=T_test, plt=plt, ax=ax)
    print('Done.')

if __name__ == '__main__':
    main()
