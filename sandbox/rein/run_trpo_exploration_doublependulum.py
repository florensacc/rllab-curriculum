import os
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import NormalizedEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from sandbox.rein.algos.trpo_unn import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
import itertools

stub(globals())

# Param ranges
seeds = range(10)
etas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
replay_pools = [True]
kl_ratios = [False]
reverse_kl_regs = [True]
param_cart_product = itertools.product(
    reverse_kl_regs, kl_ratios, replay_pools, etas, seeds
)

for reverse_kl_reg, kl_ratio, replay_pool, eta, seed in param_cart_product:

    mdp_class = DoublePendulumEnv
    mdp = NormalizedEnv(env=mdp_class())

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
        whole_paths=False,
        max_path_length=100,
        n_itr=500,
        step_size=0.01,
        eta=eta,
        eta_discount=0.998,
        snn_n_samples=10,
        subsample_factor=1.0,
        use_reverse_kl_reg=reverse_kl_reg,
        use_replay_pool=replay_pool,
        use_kl_ratio=kl_ratio,
        n_itr_update=5,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="doublependulum",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="lab_kube",
        dry=False,
    )
