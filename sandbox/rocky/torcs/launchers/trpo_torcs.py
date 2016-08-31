from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.envs.normalized_env import normalize
from sandbox.rocky.torcs.envs.torcs_env import TorcsEnv

from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

stub(globals())

from rllab.misc.instrument import VariantGenerator

env = normalize(TorcsEnv())
policy = GaussianMLPPolicy(env_spec=env.spec)
baseline = LinearFeatureBaseline(env_spec=env.spec)

vg = VariantGenerator()

vg.add("batch_size", [500, 5000, 50000])
vg.add("max_path_length", [500, 5000, 50000])
vg.add("seed", [11, 21, 31, 41, 51])

for v in vg.variants():
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=v["batch_size"],
        max_path_length=v["max_path_length"],
        discount=0.99,
    )

    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 0.8,
        },
        "limits": {
            "cpu": 0.8,
        },
    }

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        seed=v["seed"],
        exp_prefix="trpo_torcs_1",
        mode="lab_kube",
        snapshot_mode="last",
        variant=v,
    )
