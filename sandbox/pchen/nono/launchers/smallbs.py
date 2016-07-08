from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.npo import NPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.diag_npg.optimizers.diagonal_natural_gradient_optimizer import DiagonalNaturalGradientOptimizer
from sandbox.pchen.nono.optimizers.cg_with_history import ConjugateGradientOptimizerWithHistory

from sandbox.pchen.poga.algos.poga import POGA
stub(globals())

env = normalize(
    CartpoleEnv()
    # SwimmerEnv()
)

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [123, 42, 998,]


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    # baseline = LinearFeatureBaseline(env_spec=env.spec)
    baseline = ZeroBaseline(env_spec=env.spec)

    # inner_algo = TRPO(
    #     env=env,
    #     policy=policy,
    #     baseline=baseline,
    #     batch_size=v.outer_bs,
    #     max_path_length=100,
    #     n_itr=40,
    #     discount=0.99,
    #     step_size=v.inner_kl,
    # )
    # algo = POGA(
    #     env=env,
    #     policy=policy,
    #     baseline=baseline,
    #     batch_size=v.outer_bs,
    #     max_path_length=100,
    #     n_itr=30,
    #     discount=0.99,
    #     inner_algo=inner_algo,
    #     optimizer=FirstOrderOptimizer(
    #         max_epochs=1,
    #         batch_size=64,
    #         learning_rate=1e-3,
    #     ),
    #     inner_n_itr=v.inner_nitr,
    #     best_ratio=v.best_ratio,
    #     # plot=True,
    # )

    budget = 500000
    bs = 200
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=bs,
        max_path_length=100,
        n_itr=budget/bs,
        discount=0.99,
        step_size=0.01,
        plot=True,
        optimizer=ConjugateGradientOptimizerWithHistory(
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=1.,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False,
            history_size=5000,
        ),
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=3,
        snapshot_mode="last",
        seed=v.seed,
        mode="local",
        plot=True,
        exp_prefix="short_poga_poke",
        # resources=dict(
        #     requests=dict(
        #         cpu=1.1,
        #     ),
        #     limits=dict(
        #         cpu=1.1,
        #     )
        # )
    )

