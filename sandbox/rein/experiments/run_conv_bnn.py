import os
from sandbox.rein.dynamics_models.bnn.run_conv_bnn import Experiment
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME
import lasagne
import numpy as np
import itertools

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

num_epochs = 5000
batch_size = 8
ind_softmax = False
num_bins = 10
pred_delta = True
dropout = False
n_batches = int(np.ceil(100 / float(batch_size)))

lst_num_train_samples = [1]
lst_learning_rate = [3e-3]
lst_batch_norm = [True]

param_cart_product = itertools.product(
    lst_num_train_samples, lst_learning_rate, lst_batch_norm
)

for num_train_samples, learning_rate, batch_norm in param_cart_product:
    bnn = ConvBNNVIME(
        state_dim=(1, 42, 42),
        action_dim=(2,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(1, 1),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='convolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='reshape',
                 shape=([0], -1)),
            dict(name='gaussian',
                 n_units=64,
                 matrix_variate_gaussian=False,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=64,
                 matrix_variate_gaussian=False,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='hadamard',
                 n_units=64,
                 matrix_variate_gaussian=False,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=64,
                 matrix_variate_gaussian=False,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='split',
                 n_units=64,
                 matrix_variate_gaussian=False,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=400,
                 matrix_variate_gaussian=False,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='reshape',
                 shape=([0], 16, 5, 5)),
            dict(name='deconvolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='deconvolution',
                 n_filters=16,
                 filter_size=(6, 6),
                 stride=(1, 1),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='deconvolution',
                 n_filters=1,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.linear,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
        ],
        n_batches=n_batches,
        batch_size=batch_size,
        n_samples=1,
        num_train_samples=num_train_samples,
        prior_sd=0.05,
        update_likelihood_sd=False,
        learning_rate=learning_rate,
        use_local_reparametrization_trick=True,
        likelihood_sd_init=0.1,
        output_type=ConvBNNVIME.OutputType.REGRESSION,
        surprise_type=ConvBNNVIME.SurpriseType.BALD,
        disable_variance=False,
        second_order_update=False,
        debug=True,
        # ---
        ind_softmax=ind_softmax,
        num_classes=num_bins,
        disable_act_rew_paths=False
    )

    e = Experiment(model=bnn,
                   ind_softmax=ind_softmax,
                   num_epochs=num_epochs,
                   pred_delta=pred_delta,
                   num_bins=num_bins)

    run_experiment_lite(
        e.main(),
        exp_prefix="conv_bnn_delta_a",
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
