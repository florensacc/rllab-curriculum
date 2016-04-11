from __future__ import print_function
from __future__ import absolute_import

from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.hierarchical_grid_world_env import HierarchicalGridWorldEnv
# from sandbox.rocky.hrl.batch_hrl import BatchHRL
# from sandbox.rocky.hrl.
from rllab.algos.trpo import TRPO
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline


stub(globals())

env = HierarchicalGridWorldEnv(
    high_grid=[
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ],
    low_grid=[
        "SFFF",
        "FFFF",
        "FFFF",
        "FFFG",
    ],
)

policy = CategoricalMLPPolicy(env_spec=env.spec)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
)

run_experiment_lite(
    algo.train(),
    snapshot_mode="last",
)

