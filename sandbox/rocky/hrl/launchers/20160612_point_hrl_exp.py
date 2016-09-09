


from sandbox.rocky.hrl.envs.point_grid_env import PointGridEnv
from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()

MAPS = [
    [
        "So..",
        ".G..",
        ".oo.",
        "....",
    ],
    [
        "So..",
        ".G..",
        ".oo.",
        "....",
    ],
    [
        "So..",
        "....",
        ".oo.",
        ".G..",
    ],
    [
        "So..",
        "....",
        ".oo.",
        "...G",
    ],
]

vg.add("map", MAPS)
vg.add("seed", [11, 111, 211, 311, 411])
vg.add("speed", [0.1, 0.05, 0.2, 0.3])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    env = PointGridEnv(desc=v["map"], speed=v["speed"])
    policy = GaussianMLPPolicy(env_spec=env.spec)
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=20000,
        max_path_length=500,
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="point_hrl_exp",
        snapshot_mode="last",
        n_parallel=3,
        seed=v["seed"],
        mode="lab_kube",
    )
