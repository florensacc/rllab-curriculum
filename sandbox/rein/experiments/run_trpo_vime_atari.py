from sandbox.rein.algos.trpo_vime import TRPO
import os
from sandbox.rein.envs.gym_env_downscaled import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME
from rllab.core.network import ConvNetwork
from sandbox.rein.envs.atari import AtariEnvX
import lasagne

from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from sandbox.rein.algos.batch_polopt_vime import BatchPolopt

stub(globals())

RECORD_VIDEO = True
num_seq_frames = 4

os.environ["THEANO_FLAGS"] = "device=gpu"

# Param ranges
seeds = range(5)
etas = [0.1]
lst_factor = [3]
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
        mdp.spec,
        num_seq_inputs=num_seq_frames,
        regressor_args=dict(
            mean_network=network,
            batchsize=30000,
            subsample_factor=0.1),
    )

    batch_norm = True
    dropout = False
    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        whole_paths=True,
        max_path_length=1500,
        n_itr=400,
        step_size=0.01,
        optimizer_args=dict(
            num_slices=30,
            subsample_factor=0.1),
        # -------------

        # VIME settings
        # -------------
        eta=eta,
        snn_n_samples=1,
        num_train_samples=1,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=kl_ratio,
        kl_batch_size=8,
        num_sample_updates=1,  # Every sample in traj batch will be used in `num_sample_updates' updates.
        normalize_reward=False,
        replay_kl_schedule=0.98,
        n_itr_update=1,  # Fake itr updates in sampler
        dyn_pool_args=dict(
            enable=True,
            size=300000,
            min_size=10,
            batch_size=32
        ),
        second_order_update=True,
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=16 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='convolution',
                 n_filters=16 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=False),
            dict(name='convolution',
                 n_filters=16 * factor,
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
            dict(name='gaussian',
                 n_units=128 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='hadamard',
                 n_units=128 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=128 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='split',
                 n_units=64,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=False),
            dict(name='gaussian',
                 n_units=1600 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='reshape',
                 shape=([0], 16 * factor, 10, 10)),
            dict(name='deconvolution',
                 n_filters=16 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='deconvolution',
                 n_filters=16 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=False),
            dict(name='deconvolution',
                 n_filters=16 * factor,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.linear,
                 batch_norm=True,
                 dropout=False,
                 deterministic=False),
        ],
        unn_learning_rate=0.001,
        surprise_transform=BatchPolopt.SurpriseTransform.CAP99PERC,
        update_likelihood_sd=False,
        output_type=ConvBNNVIME.OutputType.CLASSIFICATION,
        likelihood_sd_init=0.1,
        prior_sd=0.05,
        predict_delta=pred_delta,
        num_seq_frames=num_seq_frames,
        # -------------
        disable_variance=False,
        surprise_type=ConvBNNVIME.SurpriseType.INFGAIN,
        predict_reward=True,
        use_local_reparametrization_trick=True,
        # -------------
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-atari-84x84-a",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
