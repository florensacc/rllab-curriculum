import os
import lasagne
import itertools

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.network import ConvNetwork
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer


from sandbox.rein.envs.atari import AtariEnvX
from sandbox.rein.algos.trpo_count import TRPO
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

# global params
num_seq_frames = 4
batch_norm = True
dropout = False

# Param ranges
seeds = list(range(5))
etas = [1., 0.01]
lst_factor = [1]
lst_pred_delta = [False]
kl_ratios = [False]
mdps = [AtariEnvX(game='freeway', obs_type="image", frame_skip=4),
        AtariEnvX(game='breakout', obs_type="image", frame_skip=4),
        AtariEnvX(game='frostbite', obs_type="image", frame_skip=4),
        AtariEnvX(game='montezuma_revenge', obs_type="image", frame_skip=4)]

param_cart_product = itertools.product(
    lst_pred_delta, lst_factor, kl_ratios, mdps, etas, seeds
)

for pred_delta, factor, kl_ratio, mdp, eta, seed in param_cart_product:
    network = ConvNetwork(
        input_shape=(num_seq_frames,) + (mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2]),
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(64,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        num_seq_inputs=num_seq_frames,
        prob_network=network,
    )

    network = ConvNetwork(
        input_shape=(num_seq_frames,) + (mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2]),
        output_dim=1,
        hidden_sizes=(64,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    baseline = GaussianMLPBaseline(
        env_spec=mdp.spec,
        num_seq_inputs=num_seq_frames,
        regressor_args=dict(
            mean_network=network,
            batchsize=None,
            subsample_factor=0.1,
            optimizer=FirstOrderOptimizer(
                max_epochs=100,
                verbose=True,
            ),
            use_trust_region=False,
        ),
    )

    # If we don't use a replay pool, we could have correct values here, as
    # it is purely Bayesian. We then divide the KL divergence term by the
    # number of batches in each iteration `batch'. Also the batch size
    # would be given correctly.
    batch_size = 1
    n_batches = 50

    autoenc = ConvBNNVIME(
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=64 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='convolution',
                 n_filters=64 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='convolution',
                 n_filters=64 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='reshape',
                 shape=([0], -1)),
            dict(name='gaussian',
                 n_units=128 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            # dict(name='discrete_embedding',
            #      n_units=128 * factor,
            #      matrix_variate_gaussian=False,
            #      nonlinearity=lasagne.nonlinearities.rectify,
            #      batch_norm=batch_norm,
            #      dropout=dropout,
            #      deterministic=False),
            dict(name='gaussian',
                 n_units=1536 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='reshape',
                 shape=([0], 64 * factor, 6, 4)),
            dict(name='deconvolution',
                 n_filters=64 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='deconvolution',
                 n_filters=64 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 0),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='deconvolution',
                 n_filters=64 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 1),
                 nonlinearity=lasagne.nonlinearities.linear,
                 batch_norm=True,
                 dropout=False,
                 deterministic=False),
        ],
        n_batches=n_batches,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=batch_size,
        n_samples=5,
        num_train_samples=1,
        prior_sd=0.05,
        second_order_update=False,
        learning_rate=0.001,
        surprise_type=ConvBNNVIME.SurpriseType.VAR,
        update_prior=False,
        update_likelihood_sd=False,
        output_type=ConvBNNVIME.OutputType.CLASSIFICATION,
        num_classes=64,
        use_local_reparametrization_trick=True,
        likelihood_sd_init=0.1,
        disable_variance=False,
        ind_softmax=True,
        num_seq_inputs=num_seq_frames,
        label_smoothing=0.003,
        disable_act_rew_paths=True  # Disable prediction of rewards and intake of actions, act as actual autoenc
    )

    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        autoenc=autoenc,
        batch_size=500,
        whole_paths=True,
        max_path_length=45,
        n_itr=400,
        step_size=0.01,
        optimizer_args=dict(
            num_slices=30,
            subsample_factor=0.1),

        # VIME settings
        # -------------
        eta=eta,
        num_sample_updates=1,  # Every sample in traj batch will be used in `num_sample_updates' updates.
        normalize_reward=False,
        replay_kl_schedule=0.98,
        n_itr_update=1,  # Fake itr updates in sampler
        dyn_pool_args=dict(
            size=100000,
            min_size=10,
            batch_size=32
        ),
        surprise_transform=None,  # BatchPolopt.SurpriseTransform.CAP99PERC,
        predict_delta=pred_delta,
        num_seq_frames=num_seq_frames,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-count-atari-42x52-a",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
