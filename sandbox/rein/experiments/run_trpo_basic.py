import os
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from sandbox.rein.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from rllab.envs.gym_env import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv

from rllab.algos.trpo import TRPO
from sandbox.john.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(3)
# mdp_classes = [MountainCarEnv]
# mdps = [NormalizedEnv(env=mdp_class()) for mdp_class in mdp_classes]
mdps = [GymEnv("MontezumaRevenge-ram-v0")]
# mdps = [GymEnv("Reacher-v1", record_video=False)]
param_cart_product = itertools.product(
    mdps, seeds
)

for mdp, seed in param_cart_product:

    #     policy = GaussianMLPPolicy(
    #         env_spec=mdp.spec,
    #         hidden_sizes=(64, 64),
    #     )

    policy = CategoricalMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 64)
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(64, 64)),
    )

    batch_size = 50000
    algo = TRPO(
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=5000,
        n_itr=500,
        step_size=0.01,
        subsample_factor=1.0,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-montezuma-a",
        n_parallel=8,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
    )
