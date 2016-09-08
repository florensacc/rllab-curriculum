


from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = TfEnv(MountainCarEnv())

policy = GaussianMLPPolicy(name="policy", env_spec=env.spec)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    max_path_length=100,
    batch_size=10000,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

run_experiment_lite(
    algo.train(),
    exp_prefix="rlrl-mc-single",
    mode="local",
    n_parallel=4,
)

