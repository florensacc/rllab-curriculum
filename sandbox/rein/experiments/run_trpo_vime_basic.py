import os
from sandbox.rein.envs.mountain_car_env_x import MountainCarEnvX
from sandbox.rein.envs.double_pendulum_env_x import DoublePendulumEnvX
from sandbox.rein.envs.cartpole_swingup_env_x import CartpoleSwingupEnvX
from sandbox.rein.envs.half_cheetah_env_x import HalfCheetahEnvX
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.envs.gym_env import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy


from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rein.algos.trpo_vime import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(10)
etas = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
normalize_rewards = [False]
kl_ratios = [True]
mdp_classes = [MountainCarEnv]
mdps = [NormalizedEnv(mdp()) for mdp in mdp_classes]

# seeds = range(10)
etas = [0.1]
# normalize_rewards = [False]
# kl_ratios = [True]
# mdps = [GymEnv("Reacher-v1", record_video=False)]
# mdps = [GymEnv("SpaceInvaders-v0")]

param_cart_product = itertools.product(
    kl_ratios, normalize_rewards, mdps, etas, seeds
)

for kl_ratio, normalize_reward, mdp, eta, seed in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(64, 64),
    )

#     policy = CategoricalMLPPolicy(
#         env_spec=mdp.spec,
#         hidden_sizes=(64, 64)
#     )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(64, 64)),
    )

    batch_size = 20
    algo = TRPO(
        discount=0.995,
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        whole_paths=True,
        max_path_length=10,
        n_itr=500,
        step_size=0.01,
        eta=eta,
        snn_n_samples=20,
        subsample_factor=1.0,
        use_replay_pool=False,
        use_kl_ratio=kl_ratio,
        use_kl_ratio_q=False,
        n_itr_update=1, # for SGD set to 20
        kl_batch_size=1,
        normalize_reward=normalize_reward,
        replay_pool_size=100000,
        n_updates_per_sample=500,
        second_order_update=False,
        unn_n_hidden=[64, 64],
        unn_layers_type=['gaussian', 'gaussian', 'gaussian'],
        unn_learning_rate=0.01,
        surprise_transform=None,  # 'cap90perc',
        pool_batch_size=10,
        update_likelihood_sd=True
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo-reacher-a",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
        script="sandbox/rein/experiments/run_experiment_lite.py",
    )
