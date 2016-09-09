from .btstrp_nn import BootstrapNetwork
from sandbox.rein.dynamics_models.bnn.utils import load_dataset_1Dregression
import lasagne
import numpy as np

def main():

    num_epochs = 1000
    batch_size = 10

    print("Loading data ...")
    (X_train, T_train), (X_test, T_test) = load_dataset_1Dregression()
    n_batches = int(np.ceil(len(X_train) / float(batch_size)))
    print("Building model and compiling functions ...")
    btstrp_nn = BootstrapNetwork(
        n_in=4,
        n_hidden=[32],
        n_out=1,
        n_batches=n_batches,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        type='regression',
        n_networks=20
    )
    
    # Build symbolic network architecture.
    btstrp_nn.build_network()
    # Build all symbolic stuff around architecture, e.g., loss, prediction
    # functions, training functions,...
    btstrp_nn.build_model()
    # Train the model.
    results = btstrp_nn.train(num_epochs=num_epochs, X_train=X_train,
                        T_train=T_train, X_test=X_test, T_test=T_test)
    print('Done.')
    
if __name__ == '__main__':
    main()