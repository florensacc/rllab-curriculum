import itertools
import os

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rein.algos.ganx_th.theano_atari import AtariEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.env_spec import EnvSpec
from rllab.spaces.box import Box
from sandbox.rein.algos.ganx_th.trpo_plus import TRPOPlus

stub(globals())

n_seq_frames = 1
model_batch_size = 32
exp_prefix = 'trpo-ganx-a'
seeds = [0]
etas = [0.001]
mdps = [AtariEnv(game='freeway', obs_type="ram+image", frame_skip=4),
        AtariEnv(game='breakout', obs_type="ram+image", frame_skip=4),
        AtariEnv(game='frostbite', obs_type="ram+image", frame_skip=4),
        AtariEnv(game='montezuma_revenge', obs_type="ram+image", frame_skip=4)]
trpo_batch_size = 50000
max_path_length = 4500

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

    algo = TRPOPlus(
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

        bonus_evaluator_args=dict(
            observation_shape=(mdp.observation_space.flat_dim,),
            observation_dtype="uint8",
            model_pool_args=dict(
                size=5000000,
                min_size=model_batch_size,
                batch_size=model_batch_size,
                subsample_factor=1,
                fill_before_subsampling=False,
            ),
        ),
        eta=eta,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix=exp_prefix,
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        use_gpu=True,
        script="sandbox/rein/algos/embedding_theano/run_experiment_lite_ram_img.py",
        # Sync ever 1h.
        periodic_sync_interval=60 * 60,
        sync_all_data_node_to_s3=True
    )
