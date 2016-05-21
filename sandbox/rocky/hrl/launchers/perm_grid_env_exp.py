from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv

stub(globals())

for grid_size in [5, 7, 9, 11]:

    for batch_size in [4000, 10000, 20000]:

        for seed in [11, 111, 211, 311, 411]:

            env = PermGridEnv(size=grid_size, n_objects=grid_size, object_seed=0)
            policy = CategoricalMLPPolicy(env_spec=env.spec)
            baseline = LinearFeatureBaseline(env_spec=env.spec)
            algo = TRPO(
                env=env,
                policy=policy,
                baseline=baseline,
                batch_size=batch_size,
                step_size=0.01,
                max_path_length=100,
                n_itr=100,
            )

            run_experiment_lite(
                algo.train(),
                exp_prefix="perm_stress_test",
                n_parallel=20,
                seed=seed,
            )
