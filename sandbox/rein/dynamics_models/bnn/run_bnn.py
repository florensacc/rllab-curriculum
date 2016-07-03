import numpy as np
from sandbox.rein.dynamics_models.bnn.bnn import BNN
from sandbox.rein.dynamics_models.utils import load_dataset_1Dregression, sliding_mean, iterate_minibatches
import lasagne
import time

# Plotting params.
# ----------------
PLOT_WEIGHTS_INDIVIDUAL = False
PLOT_WEIGHTS_TOTAL = False
PLOT_OUTPUT = True
PLOT_OUTPUT_REGIONS = False
PLOT_KL = False

def train(model, num_epochs=500, X_train=None, T_train=None, X_test=None, T_test=None):

    training_data_start = 1000
    training_data_end = 1100

    print('Training ...')

    kl_div_means = []
    kl_div_stdns = []
    kl_all_values = []

    # Plot functionality
    # ------------------
    # Print weights from this layer.
    layer = lasagne.layers.get_all_layers(model.network)[-1]
    if PLOT_WEIGHTS_TOTAL:
        import matplotlib.pyplot as plt
        plt.ion()
        sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
        mean = layer.mu.eval().ravel()
        painter_weights_total, = plt.plot(
            mean, sd, 'o', color=(1.0, 0, 0, 0.5))
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
        painter_weights_individual = []
        for i in xrange(n_plots_v):
            for j in xrange(n_plots_h):
                hl, = axarr[i][j].plot(x, x)
                axarr[i][j].set_ylim(ymin=0, ymax=2)
                painter_weights_individual.append(hl)
        plt.draw()
        plt.show()

    elif PLOT_OUTPUT:
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
        # For Jupyter notebook
        try:
            mng = plt.get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        except:
            pass
        plt.show()

    elif PLOT_KL:
        import matplotlib.pyplot as plt
        plt.ion()
        _, ax = plt.subplots(1)
        painter_kl, = ax.plot([], [], label="y", color=(1, 0, 0, 1))
        plt.xlim(xmin=0 * model.n_batches, xmax=100)
        plt.ylim(ymin=0, ymax=0.1)
        ax.grid()
        plt.draw()
        plt.show()
    # ------------------

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
            _train_err = model.train_fn(inputs, targets, 1.0)
            train_err += _train_err
            train_batches += 1

            # Calculate current minibatch KL.
            kl_mb_closed_form = model.fn_kl()

            kl_values.append(kl_mb_closed_form)
            kl_all_values.append(kl_mb_closed_form)

        # Calculate KL divergence variance over all minibatches.
        kl_mean = np.mean(np.asarray(kl_values))
        kl_stdn = np.std(np.asarray(kl_values))

        kl_div_means.append(kl_mean)
        kl_div_stdns.append(kl_stdn)

        # Plot functionality
        # ------------------
        if PLOT_WEIGHTS_TOTAL:
            sd = np.log(1 + np.exp(layer.rho.eval())).ravel()
            mean = layer.mu.eval().ravel()
            painter_weights_total.set_xdata(mean)
            painter_weights_total.set_ydata(sd)
            plt.draw()

        elif PLOT_WEIGHTS_INDIVIDUAL:
            for i in xrange(n_plots):
                w_mu = layer.mu.eval()[i, 0]
                w_rho = layer.rho.eval()[i, 0]
                w_sigma = np.log(1 + np.exp(w_rho))
                y = normal(x, w_mu, w_sigma)
                painter_weights_individual[i].set_ydata(y)
            plt.draw()

        elif PLOT_OUTPUT:
            y = [model.pred_fn(x[None, :])[0][0] for x in X_test]
            painter_output.set_ydata(y)
            plt.draw()
            plt.pause(0.00001)


        elif PLOT_OUTPUT_REGIONS and epoch % 30 == 0 and epoch != 0:
            import matplotlib.pyplot as plt

            ys = []
            for i in xrange(100):
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
                       for i in sorted(enumerate(X_test[:, 0][:, None]), key=lambda x:x[1])]
            y_mean = y_mean[indices].flatten()
            y_std = y_std[indices].flatten()
            y_median = y_median[indices].flatten()
            y_first_quart = y_first_quart[indices].flatten()
            y_third_quart = y_third_quart[indices].flatten()
            _X_test = np.array(X_test[indices][:, 0][:, None]).flatten()

            window_size = 25
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
                low=0, high=(100 - 1), size=ys.shape[0])
            axarr[0].plot(np.array(X_test)[:, 0][:, None], np.array(
                ys[range(ys.shape[0]), rnd_indices]), 'o', label="y", color=(0, 0.7, 0, 0.2))
            axarr[0].set_xlim([-8.5, 9.5])
            axarr[0].set_ylim([-5, 5])
            axarr[1].set_xlim([-8.5, 9.5])
            axarr[1].set_ylim([0, 25])
            plt.draw()
            plt.show()

        elif PLOT_KL:
            painter_kl.set_xdata(range((epoch + 1)))
            painter_kl.set_ydata(kl_div_means)
            plt.draw()
        # ------------------

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(
            train_err / train_batches))
        print(
            "  KL divergence:\t\t{:.6f} ({:.6f})".format(kl_mean, kl_stdn))
        print(model.likelihood_sd.eval())

    print("Done training.")


def main():

    num_epochs = 1000
    batch_size = 10

    print("Loading data ...")
    (X_train, T_train), (X_test, T_test) = load_dataset_1Dregression()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))
    print("Building model and compiling functions ...")
    bnn = BNN(
        n_in=4,
        n_out=1,
        n_batches=n_batches,
        n_hidden=[32],
        layers_type=['gaussian', 'gaussian'],
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=10,
        use_reverse_kl_reg=False,
        reverse_kl_reg_factor=1e-2,
        learning_rate=0.001,
        group_variance_by='unit',
        use_local_reparametrization_trick=False,
        update_likelihood_sd=False,
        likelihood_sd_init=0.01
    )
    
    bnn.fn_kl()
    print(bnn.network.get_b().eval())

    # Train the model.
    train(bnn, num_epochs=num_epochs, X_train=X_train,
              T_train=T_train, X_test=X_test, T_test=T_test)
    print('Done.')

if __name__ == '__main__':
    main()
