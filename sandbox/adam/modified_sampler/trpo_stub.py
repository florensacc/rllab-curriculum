"""
Differences from normal TRPO stub call:
1. import modified BatchSampler
2. assign modified BatchSampler as sampler_cls in TRPO initialization
3. call run_experiment_lite() with script which initializes modified sampler
"""

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.adam.modified_sampler.batch_sampler import BatchSampler

stub(globals())

env = normalize(GymEnv("Hopper-v1", record_video=False, record_log=False))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(64, 64)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    sampler_cls=BatchSampler,
    batch_size=8000,
    max_path_length=200,  # env.horizon,
    n_itr=10,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
)

run_experiment_lite(
    algo.train(),
    # Different script initializes the modified parallel sampler
    script="sandbox/adam/modified_sampler/run_experiment_lite.py",
    # Number of parallel workers for sampling
    n_parallel=2,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
