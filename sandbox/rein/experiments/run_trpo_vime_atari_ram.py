from sandbox.rein.algos.trpo_vime import TRPO
import os
from rllab.envs.gym_env import GymEnv
from sandbox.rein.dynamics_models.bnn.conv_bnn_vime import ConvBNNVIME

from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rein.dynamics_models.bnn.bnn import BNN
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.misc.instrument import stub, run_experiment_lite
import itertools
from sandbox.rein.algos.batch_polopt_vime import BatchPolopt
import lasagne

os.environ["THEANO_FLAGS"] = "device=gpu"

stub(globals())
dropout = False
batch_norm = True
factor = 1

# Param ranges
seeds = range(5)
etas = [0.001, 0.01, 0.1]
normalize_rewards = [False]
kl_ratios = [False]
RECORD_VIDEO = False
mdps = [GymEnv("Freeway-ram-v0", record_video=RECORD_VIDEO),
        GymEnv("Breakout-ram-v0", record_video=RECORD_VIDEO),
        GymEnv("Frostbite-ram-v0", record_video=RECORD_VIDEO),
        GymEnv("MontezumaRevenge-ram-v0", record_video=RECORD_VIDEO)]

param_cart_product = itertools.product(
    kl_ratios, normalize_rewards, mdps, etas, seeds
)

for kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:
    policy = CategoricalMLPPolicy(env_spec=mdp.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=mdp.spec)

    algo = TRPO(
        # TRPO settings
        # -------------
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=100,
        whole_paths=True,
        max_path_length=50,
        n_itr=250,
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
        kl_batch_size=4,
        num_sample_updates=3,  # Every sample in traj batch will be used in `num_sample_updates' updates.
        normalize_reward=False,
        replay_kl_schedule=0.98,
        n_itr_update=1,  # Fake itr updates in sampler
        dyn_pool_args=dict(
            enable=True,
            size=100000,
            min_size=10,
            batch_size=32
        ),
        second_order_update=True,
        state_dim=mdp.spec.observation_space.shape,
        action_dim=(mdp.spec.action_space.flat_dim,),
        reward_dim=(1,),
        layers_disc=[
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
                 n_units=32 * factor,
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
        ],
        unn_learning_rate=0.003,
        surprise_transform=BatchPolopt.SurpriseTransform.CAP99PERC,
        update_likelihood_sd=False,
        output_type=ConvBNNVIME.OutputType.CLASSIFICATION,
        likelihood_sd_init=0.1,
        prior_sd=0.05,
        predict_delta=False,
        num_seq_frames=1,
        # -------------
        disable_variance=False,
        surprise_type=ConvBNNVIME.SurpriseType.COMPR,
        predict_reward=True,
        use_local_reparametrization_trick=True,
        # -------------
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-vime-atari-ram-a",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
