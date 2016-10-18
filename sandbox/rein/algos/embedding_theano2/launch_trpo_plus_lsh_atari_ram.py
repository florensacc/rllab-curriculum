import itertools
import lasagne

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.algos.embedding_theano2.theano_atari import AtariEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rein.algos.embedding_theano2.trpo_plus_lsh import TRPOPlusLSH
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.env_spec import EnvSpec
from rllab.spaces.box import Box

stub(globals())

n_seq_frames = 1
model_batch_size = 32
exp_prefix = 'trpo-auto-test'
seeds = [0]
etas = [0.01]
mdps = [  # AtariEnv(game='freeway', obs_type="ram+image", frame_skip=4),
    # AtariEnv(game='breakout', obs_type="ram+image", frame_skip=4),
    # AtariEnv(game='frostbite', obs_type="ram+image", frame_skip=4),
    AtariEnv(game='montezuma_revenge', obs_type="ram+image", frame_skip=8)]
# AtariEnv(game='venture', obs_type="ram+image", frame_skip=4)]
# AtariEnv(game='private_eye', obs_type="ram+image", frame_skip=4)]
trpo_batch_size = 50000
max_path_length = 4500
dropout = False
batch_norm = False

param_cart_product = itertools.product(
    mdps, etas, seeds
)

for mdp, eta, seed in param_cart_product:
    mdp_spec = EnvSpec(
        observation_space=Box(low=-1, high=1, shape=(1, 128)),
        action_space=mdp.spec.action_space
    )

    policy = CategoricalMLPPolicy(
        env_spec=mdp_spec,
        hidden_sizes=(32, 32),
    )
    baseline = LinearFeatureBaseline(env_spec=mdp_spec)

    env_spec = EnvSpec(
        observation_space=Box(low=-1, high=1, shape=(n_seq_frames, 52, 52)),
        action_space=mdp.spec.action_space
    )

    # Alex' settings
    policy_opt_args = dict(
        cg_iters=10,
        reg_coeff=1e-5,
        subsample_factor=1.,
        max_backtracks=15,
        backtrack_ratio=0.8,
        accept_violation=False,
        hvp_approach=None,
        num_slices=10,
    )

    model_args = dict(
        state_dim=env_spec.observation_space.shape,
        action_dim=(env_spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='convolution',
                 n_filters=96,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='convolution',
                 n_filters=96,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(1, 1),
                 batch_norm=batch_norm,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 dropout=False,
                 deterministic=True),
            dict(name='convolution',
                 n_filters=96,
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
                 n_units=2,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=True),
            dict(name='discrete_embedding',
                 n_units=512,
                 batch_norm=batch_norm,
                 deterministic=True),
            dict(name='gaussian',
                 n_units=2,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=True),
            dict(name='gaussian',
                 n_units=2400,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='reshape',
                 shape=([0], 96, 5, 5)),
            dict(name='deconvolution',
                 n_filters=96,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(2, 2),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='deconvolution',
                 n_filters=96,
                 filter_size=(6, 6),
                 stride=(2, 2),
                 pad=(0, 0),
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=False,
                 deterministic=True),
            dict(name='deconvolution',
                 n_filters=96,
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
        n_samples=1,
        num_train_samples=1,
        prior_sd=0.05,
        second_order_update=False,
        learning_rate=0.0003,
        surprise_type=None,
        update_prior=False,
        update_likelihood_sd=False,
        output_type='classfication',
        num_classes=64,
        likelihood_sd_init=0.1,
        disable_variance=False,
        ind_softmax=True,
        num_seq_inputs=1,
        label_smoothing=0.003,
        # Disable prediction of rewards and intake of actions, act as actual autoenc
        disable_act_rew_paths=True,
        # --
        # Count settings
        # Put penalty for being at 0.5 in sigmoid postactivations.
        binary_penalty=True,
    )

    algo = TRPOPlusLSH(
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=trpo_batch_size,
        max_path_length=max_path_length,
        n_itr=2000,
        step_size=0.01,
        optimizer_args=policy_opt_args,
        n_seq_frames=n_seq_frames,
        # --
        # Count settings
        model_pool_args=dict(
            size=5000000,
            min_size=model_batch_size,
            batch_size=model_batch_size,
            subsample_factor=1,
            fill_before_subsampling=False,
        ),
        eta=eta,
        train_model=True,
        train_model_freq=5,
        continuous_embedding=False,
        model_embedding=True,
        sim_hash_args=dict(
            dim_key=64,
            bucket_sizes=None,
            disable_rnd_proj=True,
        ),
        clip_rewards=False,
        model_args=model_args,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        n_parallel=2,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/algos/embedding_theano/run_experiment_lite_ram_img.py",
        # Sync every 1h.
        periodic_sync_interval=60 * 60,
        sync_all_data_node_to_s3=True
    )
