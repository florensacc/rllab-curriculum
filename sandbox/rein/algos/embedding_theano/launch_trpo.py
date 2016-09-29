import itertools
import os
import lasagne

from rllab.envs.env_spec import EnvSpec
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.algos.embedding_theano.theano_atari import AtariEnv
from sandbox.rein.algos.embedding_theano.trpo_plus import TRPOPlus
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.network import ConvNetwork
from sandbox.rein.dynamics_models.bnn.conv_bnn_count import ConvBNNVIME
from rllab.spaces.box import Box

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

TEST_RUN = True

# global params
n_seq_frames = 4
model_batch_size = 32

# Param ranges
if TEST_RUN:
    exp_prefix = 'trpo-emb-g-notrain'
    seeds = range(5)
    etas = [0, 0.1, 0.01]
    mdps = [AtariEnv(game='freeway', obs_type="image", frame_skip=4),
            AtariEnv(game='frostbite', obs_type="image", frame_skip=4),
            AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=4),
            AtariEnv(game='breakout', obs_type="image", frame_skip=4)]
    lst_factor = [2]
    trpo_batch_size = 50000
    max_path_length = 4500
    dropout = False
    batch_norm = True
else:
    exp_prefix = 'trpo-pxlnn-a'
    seeds = range(5)
    etas = [0, 1.0, 0.1, 0.01]
    mdps = [AtariEnv(game='frostbite', obs_type="image", frame_skip=4),
            AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=4),
            AtariEnv(game='freeway', obs_type="image", frame_skip=4)]
    lst_factor = [1]
    trpo_batch_size = 20000
    max_path_length = 4500
    dropout = False
    batch_norm = True

param_cart_product = itertools.product(
    lst_factor, mdps, etas, seeds
)

for factor, mdp, eta, seed in param_cart_product:
    env_spec = EnvSpec(
        observation_space=Box(low=-1, high=1, shape=(52, 52, n_seq_frames)),
        action_space=mdp.spec.action_space
    )

    network = ConvNetwork(
        input_shape=(n_seq_frames,) + (
            mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2]),
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(128,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 6),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        num_seq_inputs=n_seq_frames,
        prob_network=network,
    )

    network = ConvNetwork(
        input_shape=(n_seq_frames,) + (
            mdp.spec.observation_space.shape[1], mdp.spec.observation_space.shape[2]),
        output_dim=1,
        hidden_sizes=(128,),
        conv_filters=(16, 16, 16),
        conv_filter_sizes=(6, 6, 16),
        conv_strides=(2, 2, 2),
        conv_pads=(0, 2, 2),
    )
    baseline = GaussianMLPBaseline(
        env_spec=mdp.spec,
        num_seq_inputs=n_seq_frames,
        regressor_args=dict(
            mean_network=network,
            subsample_factor=0.1,
            # optimizer=FirstOrderOptimizer(
            #     max_epochs=100,
            #     verbose=True,
            # ),
            use_trust_region=False,
        ),
    )

    autoenc = ConvBNNVIME(
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='convolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='convolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='reshape',
                 shape=([0], -1)),
            dict(name='gaussian',
                 n_units=128 * factor,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=True),
            dict(name='discrete_embedding',
                 n_units=256,
                 deterministic=True),
            dict(name='gaussian',
                 n_units=2304,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='reshape',
                 shape=([0], 64, 6, 6)),
            dict(name='deconvolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='deconvolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='deconvolution',
                 n_filters=64,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.linear,
                 batch_norm=True,
                 dropout=False,
                 deterministic=True),
        ],
        n_batches=1,
        trans_func=lasagne.nonlinearities.rectify,
        out_func=lasagne.nonlinearities.linear,
        batch_size=model_batch_size,
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
        num_seq_inputs=1,
        label_smoothing=0.003,
        # Disable prediction of rewards and intake of actions, act as actual autoenc
        disable_act_rew_paths=True,
        # --
        # Count settings
        # Put penalty for being at 0.5 in sigmoid preactivations.
        binary_penalty=True,
    )

    algo = TRPOPlus(
        model=autoenc,
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=trpo_batch_size,
        max_path_length=max_path_length,
        n_itr=1000,
        step_size=0.01,
        optimizer_args=dict(
            num_slices=30,
            subsample_factor=0.1,
        ),
        n_seq_frames=n_seq_frames,
        # --
        # Count settings
        model_pool_args=dict(
            size=100000,
            min_size=model_batch_size,
            batch_size=model_batch_size,
            subsample_factor=0.1,
            fill_before_subsampling=True,
        ),
        hamming_distance=0,
        eta=eta,
        train_model=True,
        train_model_freq=1000000000,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/algos/embedding_theano/run_experiment_lite.py",
        sync_all_data_node_to_s3=True
    )
