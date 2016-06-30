from __future__ import print_function
from __future__ import absolute_import

from rllab.algos.npo import NPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.pchen.diag_npg.optimizers.diagonal_natural_gradient_optimizer import DiagonalNaturalGradientOptimizer
from sandbox.rocky.new_dpg import CartpoleEnv

stub(globals())
from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [123, 42, 998]

    @variant
    def batch_size(self):
        return [5000, 50000]

    @variant
    def kl(self):
        return [
            # 0.1,
            0.01,
            0.001,
            0.0001,
        ]

    @variant
    def env(self):
        return map(normalize, [
            CartpoleEnv(),
            CartpoleSwingupEnv(),
            HalfCheetahEnv(),
            SwimmerEnv(),
        ])

    @variant
    def mode(self):
        return [
            "diag_hess",
            "block_diag_hess",
            "cg_block_diag_hess",
            "logprob_square",
            "full_fim",
            "cg_full_fim",
            "cg_full_hess",
        ]


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    env = v.env

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = NPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,
        max_path_length=500,
        n_itr=100,
        discount=0.99,
        step_size=v.kl,
        optimizer=DiagonalNaturalGradientOptimizer(
            mode=v.mode,
        ),
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=4,
        snapshot_mode="last",
        seed=v.seed,
        mode="lab_kube",
        # mode="local",
        exp_prefix="diag_sweep_2",
    )

