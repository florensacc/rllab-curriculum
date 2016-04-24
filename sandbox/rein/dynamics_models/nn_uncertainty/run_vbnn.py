import numpy as np
from sandbox.rein.dynamics_models.nn_uncertainty.vbnn import VBNN
from utils import load_dataset_1Dregression, load_dataset_MNIST
import lasagne
# import theano
# theano.config.mode='FAST_COMPILE'
# theano.config.optimizer='fast_compile'


def main():

    num_epochs = 1000
    batch_size = 1

    ###########################
    ### UNCOMMENT FOR MNIST ###
    ###########################
#     prob_nn.PLOT_OUTPUT = False
#     prob_nn.PLOT_WEIGHTS_INDIVIDUAL = False
#     prob_nn.PLOT_WEIGHTS_TOTAL = True
#     print("Loading data ...")
#     X_train, T_train, X_test, T_test = load_dataset_MNIST()
#     n_batches = np.ceil(len(X_train) / float(batch_size))
#     # Create neural network model
#     print("Building model and compiling functions ...")
#     vbnn = VBNN(
#         n_in=28 * 28,
#         n_hidden=[50],
#         n_out=10,
#         n_batches=n_batches,
#         layers_type=[1, 1],
#         trans_func=lasagne.nonlinearities.tanh,
#         out_func=lasagne.nonlinearities.softmax,
#         batch_size=batch_size,
#         n_samples=1,
#         type='classification',
#         prior_sd=0.15
#     )

    ################################
    ### UNCOMMENT FOR REGRESSION ###
    ################################
    print("Loading data ...")
    (X_train, T_train), (X_test, T_test) = load_dataset_1Dregression()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))
    print("Building model and compiling functions ...")
    vbnn = VBNN(
        n_in=4,
        n_hidden=[32],
        n_out=1,
        n_batches=n_batches,
        layers_type=[1, 1],
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=10,
        type='regression',
        prior_sd=0.05,
        use_reverse_kl_reg=True,
        reverse_kl_reg_factor=0.1,
        stochastic_output=False
    )

    # Train the model.
    vbnn.train(num_epochs=num_epochs, X_train=X_train,
              T_train=T_train, X_test=X_test, T_test=T_test)
    print('Done.')

if __name__ == '__main__':
    main()
