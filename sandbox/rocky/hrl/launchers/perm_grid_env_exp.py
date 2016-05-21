from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv

stub(globals())


env = PermGridEnv(size=5, n_objects=5, object_seed=0)
policy = CategoricalMLPPolicy(env_spec=env.spec)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    step_size=0.01,
    max_path_length=100,
)

run_experiment_lite(
    algo.train(),
    n_parallel=1,
    seed=0,

)
