


from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.envs.bimountain_car_env import BimountainCarEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp

from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

episode_env = BimountainCarEnv()
trial_env = MultiEnv(wrapped_env=episode_env, n_episodes=5, episode_horizon=100)

env = TfEnv(trial_env)

policy = GaussianGRUPolicy(name="policy", env_spec=env.spec, state_include_action=False)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    max_path_length=500,
    batch_size=10000,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)

run_experiment_lite(
    algo.train(),
    exp_prefix="rlrl-mc",
    mode="local",
    n_parallel=4,
)

