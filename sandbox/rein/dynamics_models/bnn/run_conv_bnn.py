import numpy as np
# from sandbox.rein.dynamics_models.bnn.conv_bnn import ConvBNN
import lasagne
from sandbox.rein.dynamics_models.utils import iterate_minibatches, plot_mnist_digit, load_dataset_MNIST, load_dataset_MNIST_plus
import time
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME
import theano
# theano.config.exception_verbosity='high'


def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None, plt=None, act=None, rew=None, im=None):
    # Train convolutional BNN to autoencode MNIST digits.

    print('Training ...')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

        pred_in = np.hstack(
            (X_train[0].reshape((1, 28 * 28)), act[0].reshape(1, 2)))
        pred_in = np.vstack((pred_in, pred_in))
        pred = model.pred_fn(pred_in)
        pred_s = pred[:, :28 * 28]
        pred_r = pred[:, -1]
        print(pred_s.shape)
        print(pred_r.shape)
        plot_mnist_digit(pred_s[0].reshape(28, 28), im)

        # Iterate over all minibatches and train on each of them.
        for batch in iterate_minibatches(X_train, T_train, model.batch_size, shuffle=True):

            # Fix old params for KL divergence computation.
            model.save_old_params()

            # Train current minibatch.
            inputs, _ = batch
            inputs_rs = inputs.reshape(-1, 1 * 28 * 28)
            _train_err = model.train_fn(
                inputs_rs, inputs_rs, 1.0)

            train_err += _train_err
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(
            train_err / train_batches))

    pred = model.pred_fn(inputs.reshape(-1, 1 * 28 * 28))
    plot_mnist_digit(pred[0].reshape(28, 28), im)
    print("Done training.")


def main():

    num_epochs = 1000
    batch_size = 2

    print("Loading data ...")
    X_train, T_train, X_test, T_test, act, rew = load_dataset_MNIST_plus()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    im = plt.imshow(
        X_train[0][0], cmap='gist_gray_r', vmin=0, vmax=1, interpolation='none')
    plot_mnist_digit(X_train[0][0], im)

    print("Building model and compiling functions ...")
    deconv_filters = 1
    filter_sizes = 5
    bnn = ConvBNNVIME(
        state_dim=(1, 28, 28),
        action_dim=(2,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution', n_filters=2,
                 filter_size=(filter_sizes, filter_sizes),
                 stride=(1, 1)),
            dict(name='pool', pool_size=(2, 2)),
            dict(name='reshape', shape=([0], -1)),
            dict(name='gaussian', n_units=288),
            dict(name='gaussian', n_units=24),
            dict(name='fuse'),
            dict(name='gaussian', n_units=288),
            dict(name='split', n_units=128),
            dict(name='reshape', shape=(
                [0], 2, 12, 12)),
            dict(name='upscale', scale_factor=2),
            dict(name='deconvolution', n_filters=deconv_filters,
                 filter_size=(filter_sizes, filter_sizes),
                 stride=(1, 1))
        ],
        n_batches=n_batches,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=2,
        prior_sd=0.05,
        update_likelihood_sd=False,
        learning_rate=0.001,
        group_variance_by=ConvBNNVIME.GroupVarianceBy.UNIT,
        use_local_reparametrization_trick=True,
        likelihood_sd_init=0.01,
        output_type=ConvBNNVIME.OutputType.REGRESSION,
        surprise_type=ConvBNNVIME.SurpriseType.BALD,
        num_classes=None,
        num_output_dim=None,
        disable_variance=False,
        second_order_update=True,
        debug=True
    )
#     bnn = ConvBNN(
#         input_dim=(1, 28, 28),
#         output_dim=(1, 28, 28),
#         layers_disc=[
#             dict(name='convolution', n_filters=2,
#                  filter_size=(filter_sizes, filter_sizes),
#                  stride=(1, 1)),
#             dict(name='pool', pool_size=(2, 2)),
#             dict(name='reshape', shape=([0], -1)),
#             dict(name='gaussian', n_units=288),
#             dict(name='gaussian', n_units=24),
#             dict(name='gaussian', n_units=288),
#             dict(name='reshape', shape=(
#                 [0], 2, 12, 12)),
#             dict(name='upscale', scale_factor=2),
#             dict(name='deconvolution', n_filters=deconv_filters,
#                  filter_size=(filter_sizes, filter_sizes),
#                  stride=(1, 1))
#         ],
#         n_batches=n_batches,
#         trans_func=lasagne.nonlinearities.rectify,
#         out_func=lasagne.nonlinearities.linear,
#         batch_size=batch_size,
#         n_samples=10,
#         prior_sd=0.05,
#         update_likelihood_sd=False,
#         learning_rate=0.001,
#         group_variance_by=ConvBNN.GroupVarianceBy.UNIT,
#         use_local_reparametrization_trick=True,
#         likelihood_sd_init=0.01,
#         output_type=ConvBNN.OutputType.REGRESSION,
#         surprise_type=ConvBNN.SurpriseType.BALD,
#         num_classes=None,
#         num_output_dim=None,
#         disable_variance=False,
#         second_order_update=True,
#         debug=True
#     )

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train,
          T_train=T_train, X_test=X_test, T_test=T_test, act=act, rew=rew, im=im)
    print('Done.')

if __name__ == '__main__':
    main()
