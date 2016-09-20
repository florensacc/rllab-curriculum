
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.adam.parallel.trpo import ParallelTRPO
from sandbox.adam.parallel.linear_feature_baseline import ParallelLinearFeatureBaseline


env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)


baseline = ParallelLinearFeatureBaseline(env_spec=env.spec)

algo = ParallelTRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=100,
    discount=0.99,
    step_size=0.01,
    n_parallel=4,
    set_cpu_affinity=False,
    whole_paths=False,
    # plot=True,
)

algo.train()
