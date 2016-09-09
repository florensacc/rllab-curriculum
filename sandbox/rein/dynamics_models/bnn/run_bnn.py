import numpy as np
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME
from sandbox.rein.dynamics_models.utils import load_dataset_1Dregression, sliding_mean, iterate_minibatches, \
    load_dataset_multitarget_classification, group, ungroup
import lasagne
import time

# Plotting params.
# ----------------
PLOT_WEIGHTS_INDIVIDUAL = False
PLOT_WEIGHTS_TOTAL = False
PLOT_OUTPUT = True
PLOT_OUTPUT_REGIONS = False
PLOT_KL = False


def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None, num_classes=None,
          num_output_dim=None, debug=False):
    training_data_start = 1000
    training_data_end = 1100

    print('Training ...')
    if PLOT_OUTPUT:
        import matplotlib.pyplot as plt
        plt.ion()
        y = [model.pred_fn(x[None, :])[0][0] for x in X_test]
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
        try:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        except:
            pass
        plt.show()

    # Finally, launch the training loop.
    print("Starting training ...")
    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err, train_batches, start_time, kl_values = 0, 0, time.time(), []

        # Iterate over all minibatches and train on each of them.
        for batch in iterate_minibatches(X_train, T_train, model.batch_size, shuffle=True):
            # Train current minibatch.
            inputs, targets, _ = batch
            inputs = np.hstack((inputs, inputs[:, -1][:, np.newaxis]))
            targets = np.hstack((targets, targets[:, -1][:, np.newaxis]))
            _train_err = model.train_fn(inputs, targets, 1.0)
            train_err += _train_err
            train_batches += 1

        if PLOT_OUTPUT:
            y = [model.pred_fn(x[None, :])[0][0] for x in X_test]
            painter_output.set_ydata(y)
            plt.draw()
            plt.pause(0.00001)

        elif PLOT_OUTPUT_REGIONS and epoch % 1 == 0 and epoch != 0:
            import matplotlib.pyplot as plt

            ys = []
            for i in range(100):
                y = [model.pred_fn(x[None, :])[0][0] for x in X_test]
                y = np.asarray(y)[:, None]
                ys.append(y)
            ys = np.hstack(ys)
            y_mean = np.mean(ys, axis=1)
            y_std = np.std(ys, axis=1)
            y_median = np.median(ys, axis=1)
            y_first_quart = np.percentile(ys, q=25, axis=1)
            y_third_quart = np.percentile(ys, q=75, axis=1)
            indices = [i[0]
                       for i in sorted(enumerate(X_test[:, 0][:, None]), key=lambda x: x[1])]
            y_mean = y_mean[indices].flatten()
            y_std = y_std[indices].flatten()
            y_median = y_median[indices].flatten()
            y_first_quart = y_first_quart[indices].flatten()
            y_third_quart = y_third_quart[indices].flatten()
            _X_test = np.array(X_test[indices][:, 0][:, None]).flatten()

            window_size = 25
            y_mean = sliding_mean(y_mean, window=window_size)
            y_std = sliding_mean(y_std, window=window_size)

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
                ys[list(range(ys.shape[0])), rnd_indices]), 'o', label="y", color=(0, 0.7, 0, 0.2))
            axarr[0].set_xlim([-8.5, 9.5])
            axarr[0].set_ylim([-5, 5])
            axarr[1].set_xlim([-8.5, 9.5])
            axarr[1].set_ylim([0, 25])
            plt.draw()
            plt.show()

        # Then we print the results for this epoch:
        print(("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time)))
        print(("  training loss:\t\t{:.6f}".format(
            train_err / train_batches)))
        print((model.likelihood_sd.eval()))

    print("Done training.")


def main():
    DEBUG = True

    num_epochs = 1000
    batch_size = 10

    print("Loading data ...")
    (X_train, T_train), (X_test, T_test) = load_dataset_1Dregression()

    num_output_dim = 2
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))
    print("Building model and compiling functions ...")
    bnn = ConvBNNVIME(
        state_dim=(4,),
        action_dim=(1,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='gaussian',
                 n_units=32,
                 matrix_variate_gaussian=False,
                 batch_norm=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=32,
                 matrix_variate_gaussian=False,
                 batch_norm=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='hadamard',
                 n_units=32,
                 matrix_variate_gaussian=False,
                 batch_norm=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=32,
                 matrix_variate_gaussian=False,
                 batch_norm=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='split',
                 n_units=32,
                 matrix_variate_gaussian=False,
                 batch_norm=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=1,
                 matrix_variate_gaussian=False,
                 batch_norm=False,
                 nonlinearity=lasagne.nonlinearities.linear,
                 dropout=False,
                 deterministic=False)
        ],
        n_batches=n_batches,
        batch_size=batch_size,
        n_samples=1,
        num_train_samples=1,
        prior_sd=0.05,
        update_likelihood_sd=False,
        learning_rate=0.001,
        use_local_reparametrization_trick=True,
        likelihood_sd_init=0.001,
        output_type=ConvBNNVIME.OutputType.REGRESSION,
        surprise_type=ConvBNNVIME.SurpriseType.L1,
        disable_variance=False,
        second_order_update=False,
        debug=True,
        update_prior=False,
        # ---
        ind_softmax=False
    )

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train,
          T_train=T_train, X_test=X_test, T_test=T_test, num_classes=1, num_output_dim=num_output_dim, debug=DEBUG)
    print('Done.')


if __name__ == '__main__':
    main()
