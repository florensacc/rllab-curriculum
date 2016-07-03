from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.npo import NPO
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.diag_npg.optimizers.diagonal_natural_gradient_optimizer import DiagonalNaturalGradientOptimizer

from sandbox.pchen.poga.algos.poga import POGA
stub(globals())

envs = [
    CartpoleEnv(),
MountainCarEnv(),
    SwimmerEnv(),
Walker2DEnv(),
]

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [123, 42, 998,]

    @variant
    def outer_bs(self):
        return [1000, 5000, 25000]
        # return [5000, 25000]

    @variant
    def inner_bs(self, outer_bs):
        # return [ratio*outer_bs for ratio in [
        #     0.5, 0.1
        # ]]
        # return [5000]
        return [500, 1000, 3000]

    @variant
    def inner_nitr(self):
        return [3, 9]

    @variant
    def inner_kl(self):
        return [0.01, 0.05]

    @variant
    def best_ratio(self):
        return [0.5,]

    @variant
    def env(self):
        return [normalize(e) for e in envs]




variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    env=v.env
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    # baseline = ZeroBaseline(env_spec=env.spec)

    inner_algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.inner_bs,
        max_path_length=100,
        n_itr=40,
        discount=0.99,
        step_size=v.inner_kl,
    )
    algo = POGA(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.outer_bs,
        max_path_length=100,
        n_itr=30,
        discount=0.99,
        inner_algo=inner_algo,
        optimizer=FirstOrderOptimizer(
            max_epochs=1,
            batch_size=64,
            learning_rate=1e-3,
        ),
        inner_n_itr=v.inner_nitr,
        best_ratio=v.best_ratio,
        # plot=True,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=2,
        snapshot_mode="last",
        seed=v.seed,
        mode="lab_kube",
        # plot=True,
        exp_prefix="short_poga_poke2",
        resources=dict(
            requests=dict(
                cpu=1.8,
            ),
            limits=dict(
                cpu=1.8,
            )
        )
    )

