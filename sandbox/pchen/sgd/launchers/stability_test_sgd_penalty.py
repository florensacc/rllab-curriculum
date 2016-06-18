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

    @variant
    def lr(self):
        return [0.001, 0.0001, 0.00001]

    @variant
    def max_epochs(self):
        return [1, 5, 10]

    @variant
    def sgd_batch_size(self):
        return [32, 64, 128]

    @variant
    def truncate_local_is_ratio(self):
        return [None, ]

    @variant
    def lossy_lr(self):
        return [False, True]

    @variant
    def max_penalty_itr(self):
        return [1, 5]

    @variant
    def barrier_coeff(self):
        return [0., 1e3]


variants = VG().variants()[:1]
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
    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v.batch_size,
        max_path_length=100,
        n_itr=100,
        step_size=0.01,
        discount=0.995,
        optimizer=PenaltyOptimizer(
            FirstOrderOptimizer(
                update_method=lasagne.updates.adam,
                learning_rate=v.lr,
                max_epochs=v.max_epochs,
                batch_size=v.sgd_batch_size,
                randomized=True,
            ),
            initial_penalty=1.,
            data_split=0.5,
            max_penalty=1e4,
            adapt_penalty=True,
            max_penalty_itr=v.max_penalty_itr,
            increase_penalty_factor=1.5,
            decrease_penalty_factor=1/1.5,
            barrier_coeff=v.barrier_coeff,
        ),
        truncate_local_is_ratio=v.truncate_local_is_ratio,
        lossy_lr=v.lossy_lr,
    )
    run_experiment_lite(
        algo.train(),
        exp_prefix="sgd_stability_test",
        n_parallel=2,
        snapshot_mode="last",
        mode="lab_kube",
        seed=v.seed,
        resouces=dict(
            requests=dict(
                cpu=1.8,
            ),
            limits=dict(
                cpu=1.8,
            )
        ),
    )

