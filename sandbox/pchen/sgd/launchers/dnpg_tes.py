


from functools import partial

from rllab.algos.ppo import PPO
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.normalized_env import normalize
from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer, lasagne
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.algos.trpo import TRPO
from sandbox.pchen.sgd.distributions.beefed_diag_gaussian import BeefedDiagGaussian
from sandbox.pchen.sgd.dnpg import DNPG
from sandbox.pchen.sgd.online_penalty_optimizer import OnlinePenaltyOptimizer
from sandbox.pchen.sgd.penalty_optimier import PenaltyOptimizer
from sandbox.rocky.hrl.optimizers.conjugate_gradient_optimizer import \
    ConjugateGradientOptimizer, PerlmutterHvp, FiniteDifferenceHvp

# stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [123, 42, 998, 1]

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
        return [
            # "trpo",
            "dnpg",
        ]

    @variant
    def env(self):
        return ["Hopper-v1", "Pendulum-v0", "Walker2d-v1", "Reacher-v1"]

    @variant
    def batch_size(self, algo):
        return [10000]
        sizes = [1000, 5000, 10000, 50000]
        if algo == "trpo":
            return sizes
        else:
            return [s*2 for s in sizes]

    @variant
    def max_penalty_itr(self, algo):
        if algo == "trpo":
            return [1]
        return [1, ]

    @variant
    def barrier_coeff(self, algo):
        if algo == "trpo":
            return [1]
        return [1e3]

    @variant
    def kl(self):
        return [0.01]#, 0.05, 0.01, 0.001]



variants = VG().variants()
print("#Experiments: %d" % len(variants))

for v in variants[:1]:
    # env = (GymEnv(v.env, record_video=False, record_log=False))
    env = normalize(CartpoleEnv())
    if v.normalize:
        env = normalize(env)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        init_std=v.init_std,
        dist_cls=BeefedDiagGaussian,
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
            max_path_length=100,
            n_itr=100,
            step_size=v.kl,
            discount=0.995,
        )
    else:
        algo = DNPG(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=v.batch_size,
            max_path_length=100,
            n_itr=100,
            step_size=v.kl,
            discount=0.995,
            optimizer=
                FirstOrderOptimizer(
                    update_method=lasagne.updates.adam,
                    learning_rate=1e-4,
                    max_epochs=25,
                    batch_size=128,
                    randomized=True,
                    verbose=True,
                ),
        )
    run_experiment_lite(
        algo.train(),
        exp_prefix="trpo_sgd_env_comp2",
        n_parallel=2,
        snapshot_mode="last",
        # mode="lab_kube",
        mode="local",
        seed=v.seed,
    )

