


from sandbox.rocky.hrl.policies.stochastic_gru_policy import StochasticGRUPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.hrl.envs.perm_grid_env import PermGridEnv

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("grid_size", [5, 7, 9, 11])
vg.add("batch_size", [4000, 10000, 20000])
vg.add("seed", [11, 111, 211, 311, 411])

for v in vg.variants():
    env = PermGridEnv(size=v["grid_size"], n_objects=v["grid_size"], object_seed=0)
    policy = StochasticGRUPolicy(env_spec=env.spec, n_subgoals=v["grid_size"])
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v["batch_size"],
        step_size=0.01,
        max_path_length=100,
        n_itr=100,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="hier_perm_stress_test",
        n_parallel=1,
        seed=v["seed"],
    )
