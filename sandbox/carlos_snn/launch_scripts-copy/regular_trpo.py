from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.mujoco.swimmer_env import SwimmerEnv
stub(globals())

env = normalize(SwimmerEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=500,
    n_itr=200,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

for s in [80]:
    run_experiment_lite(
        algo.train(),
        # where to launch the instances
        mode='local',
        # Number of parallel workers for sampling
        n_parallel=8,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=s,
        # plot=True,
        exp_prefix='swimmer-trpo',
        exp_name='swimmer-trpo-50kBs-200itr-500pl-{}'.format(s)
    )
