from __future__ import absolute_import
from __future__ import print_function

from functools import partial

from rllab.algos.ppo import PPO
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

# stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


# class VG(VariantGenerator):
#     @variant
#     def seed(self):
#         return [11, 111, 211, 311, 411]
#
#     @variant
#     def hvp_approach_cls(self):
#         return [PerlmutterHvp, FiniteDifferenceHvp]
#
#     @variant
#     def symmetric(self, hvp_approach_cls):
#         if hvp_approach_cls == FiniteDifferenceHvp:
#             return [True, False]
#         else:
#             return [True]
#
#     @variant
#     def base_eps(self, hvp_approach_cls):
#         if hvp_approach_cls == FiniteDifferenceHvp:
#             return [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
#         else:
#             return [1e-8]
#
#
# variants = VG().variants()
# print("#Experiments: %d" % len(variants))

# for v in variants:
#     env = HalfCheetahEnv()
#     policy = GaussianMLPPolicy(env_spec=env.spec)
#     baseline = LinearFeatureBaseline(env_spec=env.spec)
#
#     if v["hvp_approach_cls"] == PerlmutterHvp:
#         hvp_approach = PerlmutterHvp()
#     elif v["hvp_approach_cls"] == FiniteDifferenceHvp:
#         hvp_approach = FiniteDifferenceHvp(base_eps=v["base_eps"], symmetric=v["symmetric"])
#     else:
#         raise NotImplementedError
#
#     algo = TRPO(
#         env=env,
#         policy=policy,
#         baseline=baseline,
#         batch_size=20000,
#         max_path_length=500,
#         n_itr=500,
#         step_size=0.01,
#         optimizer=ConjugateGradientOptimizer(hvp_approach=hvp_approach),
#     )
#
#     run_experiment_lite(
#         algo.train(),
#         exp_prefix="hvp_exp",
#         n_parallel=3,
#         snapshot_mode="last",
#         mode="lab_kube",
#         seed=v["seed"],
#     )

# env = normalize(HopperEnv())
env = (GymEnv("Hopper-v1"))
policy = GaussianMLPPolicy(
    env_spec=env.spec,
    init_std=1.,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = PPO(
    env=env,
    policy=policy,
    baseline=baseline,
    # batch_size=50000,
    batch_size=10000,
    max_path_length=100,
    n_itr=500,
    step_size=0.01,
    discount=0.995,
    # step_size=2.,
    optimizer=PenaltyOptimizer(
        FirstOrderOptimizer(
            update_method=partial(lasagne.updates.adam, learning_rate=1e-3),
            # update_method=partial(lasagne.updates.rmsprop, learning_rate=1e-4),
            # max_epochs=1,
            max_epochs=10,
            batch_size=128,
        ),
        data_split=0.5,
        max_penalty=1e5,
        adapt_penalty=True,
        adapt_itr=10,
        max_penalty_itr=3,

        # initial_penalty=0.,
        # min_penalty=0.,
    ),
    # optimizer_args=dict(
    #     max_opt_itr=100,
    #     decrease_penalty_factor=0.8,
    # )
    # truncate_local_is_ratio=10.,
    lossy_lr=False,
)
run_experiment_lite(
    algo.train(),
    exp_prefix="sgd_test",
    n_parallel=3,
    snapshot_mode="last",
    mode="local",
    seed=422,
)
