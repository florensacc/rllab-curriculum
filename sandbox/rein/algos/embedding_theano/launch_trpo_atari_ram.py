import itertools
import os
import lasagne

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.algos.embedding_theano.theano_atari import AtariEnv
from sandbox.rein.algos.embedding_theano.trpo_plus import TRPOPlus
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from sandbox.rein.dynamics_models.bnn.conv_bnn_count import ConvBNNVIME
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())

n_seq_frames = 1
model_batch_size = 32
exp_prefix = 'trpo-ram-emb-a'
seeds = range(1)
etas = [0.1, 0.01]
mdps = [AtariEnv(game='freeway', obs_type="ram", frame_skip=8),
        AtariEnv(game='breakout', obs_type="ram", frame_skip=4),
        AtariEnv(game='frostbite', obs_type="ram", frame_skip=8),
        AtariEnv(game='montezuma_revenge', obs_type="ram", frame_skip=8)]
trpo_batch_size = 500
max_path_length = 45
dropout = False
batch_norm = True

param_cart_product = itertools.product(
    mdps, etas, seeds
)

for mdp, eta, seed in param_cart_product:
    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32, 32),
    )
    baseline = LinearFeatureBaseline(env_spec=mdp.spec)

    autoenc = ConvBNNVIME(
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
            dict(name='gaussian',
                 n_units=1024,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
                 deterministic=True),
            dict(name='discrete_embedding',
                 n_units=512,
                 deterministic=True),
            dict(name='gaussian',
                 n_units=1024,
                 matrix_variate_gaussian=False,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 batch_norm=batch_norm,
                 dropout=dropout,
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
        # Put penalty for being at 0.5 in sigmoid postactivations.
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
        n_itr=500,
        step_size=0.01,
        optimizer_args=dict(
            num_slices=50,
            subsample_factor=0.1,
        ),
        n_seq_frames=n_seq_frames,
        # --
        # Count settings
        model_pool_args=dict(
            size=1000000,
            min_size=model_batch_size,
            batch_size=model_batch_size,
            subsample_factor=0.3,
            fill_before_subsampling=False,
        ),
        hamming_distance=0,
        eta=eta,
        train_model=False,
        train_model_freq=5,
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
        script="sandbox/rein/algos/embedding_theano/run_experiment_lite.py",
        sync_all_data_node_to_s3=True
    )
