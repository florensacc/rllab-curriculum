
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

from sandbox.adam.parallel.trpo import ParallelTRPO
from sandbox.adam.parallel.linear_feature_baseline import ParallelLinearFeatureBaseline

stub(globals())


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
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    n_parallel=4,
    set_cpu_affinity=False,  # (need package psutil if True)
    whole_paths=False,
    # plot=True,
)

run_experiment_lite(
    algo.train(),
    script="sandbox/adam/run_experiment_lite_par.py",
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
