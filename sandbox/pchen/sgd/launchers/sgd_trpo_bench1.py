from __future__ import absolute_import
from __future__ import print_function

from functools import partial

from rllab.algos.ppo import PPO
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
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
        return [123, 42, 998, 1, 998]

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
    def algo(self):
        return ["trpo", "sgd"]

    @variant
    def env(self):
        return [
            CartpoleSwingupEnv(),
            InvertedDoublePendulumEnv(),
            SwimmerEnv(),
            AntEnv(),
        ]
        # return ["Hopper-v1", "Pendulum-v0", "Walker2d-v1", "Reacher-v1"]

    @variant
    def batch_size(self, algo):
        sizes = [50000]
        if algo == "trpo":
            return sizes
        else:
            return [s*2 for s in sizes]

    @variant
    def max_penalty_itr(self, algo):
        if algo == "trpo":
            return [1]
        return [3, ]

    @variant
    def barrier_coeff(self, algo):
        if algo == "trpo":
            return [1]
        return [0., 1e2, 1e4]

    @variant
    def kl(self):
        return [0.1, 0.05, 0.01, 0.001]



variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants:
    # env = (GymEnv(v.env, record_video=False, record_log=False))
    env = v["env"]
    if v.normalize:
        env = normalize(env)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        init_std=v.init_std,
        hidden_sizes=[100,50,25],
    )
    if v.baseline == "linear":
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    elif v.baseline == "zero":
        baseline = ZeroBaseline(env_spec=env.spec)
    else:
        raise 123
    if v.algo == "trpo":
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v.batch_size,
            max_path_length=500,
            n_itr=500,
            step_size=v.kl,
            discount=0.99,
        )
    else:
        algo = PPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v.batch_size,
            max_path_length=500,
            n_itr=500,
            step_size=v.kl,
            discount=0.99,
            optimizer=PenaltyOptimizer(
                FirstOrderOptimizer(
                    update_method=lasagne.updates.adam,
                    learning_rate=1e-3,
                    max_epochs=3,
                    batch_size=128,
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
        )
    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo_sgd_bench1",
        n_parallel=2,
        snapshot_mode="last",
        mode="lab_kube",
        seed=v.seed,
    )

