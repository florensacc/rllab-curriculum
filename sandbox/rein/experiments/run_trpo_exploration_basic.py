import os
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rein.algos.trpo_unn import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(10)
etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
kl_ratios = [False]
reverse_kl_regs = [True]
normalize_rewards = [False]
mdp_classes = [CartpoleEnv, CartpoleSwingupEnv,
               DoublePendulumEnv, MountainCarEnv]
mdps = [NormalizedEnv(env=mdp_class()) for mdp_class in mdp_classes]
param_cart_product = itertools.product(
    mdps, reverse_kl_regs, kl_ratios, etas, seeds, normalize_rewards
)

for mdp, reverse_kl_reg, kl_ratio, eta, seed, normalize_reward in param_cart_product:

    policy = GaussianMLPPolicy(
        env_spec=mdp.spec,
        hidden_sizes=(32,),
    )

    baseline = GaussianMLPBaseline(
        mdp.spec,
        regressor_args=dict(hidden_sizes=(32,)),
    )

    algo = TRPO(
        env=mdp,
        policy=policy,
        baseline=baseline,
        batch_size=1000,
        whole_paths=True,
        max_path_length=500,
        n_itr=1000,
        step_size=0.01,
        eta=eta,
        eta_discount=1.0,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_reverse_kl_reg=reverse_kl_reg,
        use_replay_pool=True,
        use_kl_ratio=kl_ratio,
        n_itr_update=5,
        kl_batch_size=5,
        normalize_reward=normalize_reward
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo_exploration",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="local",
        dry=False,
    )
