from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.adam.parallel.trpo import ParallelTRPO
from sandbox.adam.parallel.linear_feature_baseline import ParallelLinearFeatureBaseline

run_parallelized = True

if not run_parallelized:
    stub(globals())


n_parallel = 4
batch_size = 4000
max_path_length = 100
n_itr = 10
whole_paths = True


env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)


if run_parallelized:
    baseline = ParallelLinearFeatureBaseline(env_spec=env.spec)

    algo = ParallelTRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=0.99,
        step_size=0.01,
        n_parallel=n_parallel,
        whole_paths=whole_paths,
        # plot=True,
    )

    algo.train()

else:
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=batch_size,
        max_path_length=max_path_length,
        n_itr=n_itr,
        discount=0.99,
        step_size=0.01,
        whole_paths=whole_paths,
        # plot=True,
    )

    run_experiment_lite(
        algo.train(),
        # Number of parallel workers for sampling
        n_parallel=n_parallel,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="none",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # plot=True,
    )
