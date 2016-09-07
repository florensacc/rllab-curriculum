import os
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from sandbox.rein.envs.gym_env_downscaled import GymEnv
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from sandbox.rein.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.core.network import ConvNetwork

os.environ["THEANO_FLAGS"] = "device=gpu"

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from sandbox.rein.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from sandbox.john.instrument import stub, run_experiment_lite
from rllab.misc.instrument import stub, run_experiment_lite

import itertools

RECORD_VIDEO = True
num_seq_frames = 4

stub(globals())

# Param ranges
seeds = list(range(10))

mdps = [GymEnv("Freeway-ram-v0", record_video=RECORD_VIDEO),
        GymEnv("Breakout-ram-v0", record_video=RECORD_VIDEO),
        GymEnv("Frostbite-ram-v0", record_video=RECORD_VIDEO),
        GymEnv("MontezumaRevenge-ram-v0", record_video=RECORD_VIDEO)]
param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:
    policy = CategoricalMLPPolicy(env_spec=mdp.spec, hidden_sizes=(32, 32))
    baseline = LinearFeatureBaseline(env_spec=mdp.spec)

    algo = TRPO(
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=100000,
        whole_paths=True,
        max_path_length=4500,
        n_itr=400,
        step_size=0.01,
        sampler_args=dict(num_seq_frames=num_seq_frames),
        optimizer_args=dict(
            num_slices=30,
            subsample_factor=0.1),
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-atari-ram-f",
        n_parallel=4,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        use_gpu=False,
        dry=False,
    )
