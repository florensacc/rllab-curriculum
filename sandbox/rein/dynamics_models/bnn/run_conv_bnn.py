import numpy as np
# from sandbox.rein.dynamics_models.bnn.conv_bnn import ConvBNN
import lasagne
from sandbox.rein.dynamics_models.utils import iterate_minibatches, plot_mnist_digit, load_dataset_MNIST, \
    load_dataset_MNIST_plus, load_dataset_Atari_plus
import time
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME


# import theano
# theano.config.exception_verbosity='high'


def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None, plt=None, act=None, rew=None,
          im=None):
    # Train convolutional BNN to autoencode MNIST digits.

    im_size = X_train.shape[-1]
    pred_in = np.hstack(
        (X_train[0].reshape((1, im_size * im_size)), act[0].reshape(1, 2)))
    pred_in = np.vstack((pred_in, pred_in))
    pred = model.pred_fn(pred_in)
    pred_s = pred[:, :im_size * im_size]
    pred_r = pred[:, -1]
    print('pred_r: {}'.format(pred_r))
    plot_mnist_digit(pred_s[0].reshape(im_size, im_size), im)

    X_train = X_train.reshape(-1, im_size * im_size)
    T_train = T_train.reshape(-1, im_size * im_size)
    X = np.hstack((X_train, act))
    Y = np.hstack((T_train, rew))

    print('Training ...')

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

        print('KL[post||prior]: {}'.format(model.log_p_w_q_w_kl().eval()))

        # Iterate over all minibatches and train on each of them.
        for batch in iterate_minibatches(X, Y, model.batch_size, shuffle=True):
            # Fix old params for KL divergence computation.
            model.save_params()

            # Train current minibatch.
            inputs, targets, _ = batch
            _train_err = model.train_fn(inputs, targets, 1.0)

            train_err += _train_err
            train_batches += 1

        pred = model.pred_fn(X)
        acc = np.mean(np.square(pred - Y))
        print('sqerr: {}'.format(acc))
        print('likelihood_sd: {}'.format(model.likelihood_sd.eval()))

        pred = model.pred_fn(X)
        pred_s = pred[0, :im_size * im_size]
        pred_r = pred[:5, -1]
        #         print(pred_r)
        #         print(Y[:5, -1])
        #         pred_s = pred_s + 1.
        plot_mnist_digit(pred_s.reshape(im_size, im_size), im)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(
            train_err / train_batches))

    print("Done training.")


def main():
    num_epochs = 1000
    batch_size = 2

    print("Loading data ...")
    X_train, T_train, act, rew = load_dataset_MNIST_plus()
    X_train, T_train, act, rew = load_dataset_Atari_plus()
    print(X_train.shape)
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))

    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()
    im = plt.imshow(
        T_train[0, 0, :, :], cmap='gist_gray_r', vmin=0, vmax=1,
        interpolation='none')
    plot_mnist_digit((X_train[0][0]), im)

    print("Building model and compiling functions ...")
    deconv_filters = 1
    filter_sizes = 4
    bnn = ConvBNNVIME(
        # state_dim=(1, 28, 28),
        # action_dim=(2,),
        # reward_dim=(1,),
        # layers_disc=[
        #     dict(name='convolution',
        #          n_filters=2,
        #          filter_size=(filter_sizes, filter_sizes),
        #          stride=(2, 2),
        #          pad=(0, 0)),
        #     dict(name='reshape',
        #          shape=([0], -1)),
        #     dict(name='gaussian',
        #          n_units=338,
        #          matrix_variate_gaussian=True),
        #     dict(name='gaussian',
        #          n_units=128,
        #          matrix_variate_gaussian=True),
        #     dict(name='hadamard',
        #          n_units=128,
        #          matrix_variate_gaussian=True),
        #     dict(name='gaussian',
        #          n_units=338,
        #          matrix_variate_gaussian=True),
        #     dict(name='split',
        #          n_units=128,
        #          matrix_variate_gaussian=True),
        #     dict(name='reshape',
        #          shape=([0], 2, 13, 13)),
        #     dict(name='deconvolution',
        #          n_filters=deconv_filters,
        #          filter_size=(filter_sizes, filter_sizes),
        #          stride=(2, 2),
        #          pad=(0, 0),
        #          nonlinearity=lasagne.nonlinearities.linear)
        # ],
        state_dim=(1, 84, 84),
        action_dim=(2,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0)),
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2)),
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2)),
            dict(name='reshape',
                 shape=([0], -1)),
            dict(name='gaussian',
                 n_units=512,
                 matrix_variate_gaussian=True),
            dict(name='gaussian',
                 n_units=512,
                 matrix_variate_gaussian=True),
            dict(name='hadamard',
                 n_units=512,
                 matrix_variate_gaussian=True),
            dict(name='gaussian',
                 n_units=512,
                 matrix_variate_gaussian=True),
            dict(name='split',
                 n_units=128,
                 matrix_variate_gaussian=True),
            dict(name='gaussian',
                 n_units=1600,
                 matrix_variate_gaussian=True),
            dict(name='reshape',
                 shape=([0], 16, 10, 10)),
            dict(name='deconvolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify),
            dict(name='deconvolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify),
            dict(name='deconvolution',
                 n_filters=1,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.linear),
        ],
        n_batches=n_batches,
        batch_size=batch_size,
        n_samples=1,
        num_train_samples=1,
        prior_sd=0.005,
        update_likelihood_sd=False,
        learning_rate=0.001,
        group_variance_by=ConvBNNVIME.GroupVarianceBy.UNIT,
        use_local_reparametrization_trick=True,
        likelihood_sd_init=0.01,
        output_type=ConvBNNVIME.OutputType.REGRESSION,
        surprise_type=ConvBNNVIME.SurpriseType.COMPR,
        num_classes=None,
        num_output_dim=None,
        disable_variance=False,
        second_order_update=False,
        debug=True,
        # ---
        ind_softmax=True
    )

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train, T_train=T_train, act=act, rew=rew, im=im)
    print('Done.')


if __name__ == '__main__':
    main()
