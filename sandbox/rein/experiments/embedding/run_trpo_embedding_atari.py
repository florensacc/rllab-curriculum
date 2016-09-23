import itertools
import os
import lasagne
import tensorflow as tf

from rllab.envs.env_spec import EnvSpec
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.algos.embedding.tf_atari import AtariEnv
from sandbox.rein.algos.embedding.trpo_plus import TRPOPlus
from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rein.dynamics_models.bnn.conv_bnn_count import ConvBNNVIME

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

TEST_RUN = True

# global params
n_seq_frames = 1

# Param ranges
if TEST_RUN:
    exp_prefix = 'debug-bin'
    seeds = range(1)
    etas = [0.1]
    mdps = [AtariEnv(game='frostbite', obs_type="image", frame_skip=8),
            AtariEnv(game='freeway', obs_type="image", frame_skip=8),
            AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=8)]
    lst_factor = [1]
    trpo_batch_size = 1000
    max_path_length = 450
    dropout = False
    batch_norm = True
else:
    exp_prefix = 'trpo-pxlnn-a'
    seeds = range(5)
    etas = [0, 1.0, 0.1, 0.01]
    mdps = [AtariEnv(game='frostbite', obs_type="image", frame_skip=8),
            AtariEnv(game='montezuma_revenge', obs_type="image", frame_skip=8),
            AtariEnv(game='freeway', obs_type="image", frame_skip=8)]
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
    # TODO: make own env_spec, feed to policy/baseline, set tf_atari correct. Make sure sampler gets correct info.
    network = ConvNetwork(
        name='policy_network',
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.softmax,
        input_shape=env_spec.observation_space.shape,  # mdp.spec.observation_space.shape,
        output_dim=mdp.spec.action_space.flat_dim,
        hidden_sizes=(64,),
        conv_filters=(16, 16),
        conv_filter_sizes=(6, 6),
        conv_strides=(2, 2),
        conv_pads=('VALID', 'VALID'),
    )
    policy = CategoricalMLPPolicy(
        name='policy',
        env_spec=env_spec,  # mdp.spec,
        prob_network=network,
    )

    network = ConvNetwork(
        name='baseline_network',
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.identity,
        input_shape=env_spec.observation_space.shape,  # mdp.spec.observation_space.shape,
        output_dim=1,
        hidden_sizes=(32,),
        conv_filters=(16, 16),
        conv_filter_sizes=(6, 6),
        conv_strides=(2, 2),
        conv_pads=('VALID', 'VALID'),
    )
    baseline = GaussianMLPBaseline(
        env_spec=env_spec,  # mdp.spec,
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
                 n_units=32,
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
        batch_size=40,
        n_samples=5,
        num_train_samples=1,
        prior_sd=0.05,
        second_order_update=False,
        learning_rate=0.003,
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
        disable_act_rew_paths=True  # Disable prediction of rewards and intake of actions, act as actual autoenc
    )

    algo = TRPOPlus(
        model=autoenc,
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=trpo_batch_size,
        max_path_length=max_path_length,
        n_itr=400,
        step_size=0.01,
        n_seq_frames=n_seq_frames,
        model_pool_args=dict(
            size=100000,
            min_size=32,
            batch_size=32,
            subsample_factor=0.1,
            fill_before_subsampling=True,
        )
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
        sync_all_data_node_to_s3=True
    )
