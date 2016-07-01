from __future__ import absolute_import
from __future__ import print_function

from functools import partial

from rllab.algos.ppo import PPO
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.normalized_env import normalize
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer, lasagne
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from sandbox.pchen.sgd.online_penalty_optimizer import OnlinePenaltyOptimizer
from sandbox.pchen.sgd.penalty_optimier import PenaltyOptimizer
from sandbox.rocky.hrl.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer, PerlmutterHvp, FiniteDifferenceHvp

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [123, 42, 998]

    @variant
    def normalize(self):
        return [True,]

    @variant
    def init_std(self):
        return [1., ]

    @variant
    def baseline(self):
        return ["linear",]

    @variant
    def batch_size(self):
        return [100000, ]


variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    env = (GymEnv("Hopper-v1", record_video=False))
    if v.normalize:
        env = normalize(env)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        init_std=v.init_std,
    )
    if v.baseline == "linear":
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif v.baseline == "zero":
        baseline = ZeroBaseline(env_spec=env.spec)
    else:
        raise 123
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,
        max_path_length=100,
        n_itr=100,
        step_size=0.01,
        discount=0.995,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="sgd_stability_test",
        n_parallel=2,
        snapshot_mode="last",
        mode="lab_kube",
        seed=v.seed,
    )

