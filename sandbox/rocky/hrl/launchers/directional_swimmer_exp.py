


from sandbox.rocky.hrl.envs.directional_swimmer_env import DirectionalSwimmerEnv
from rllab.algos.trpo import TRPO
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator

from rllab import config
config.AWS_INSTANCE_TYPE = ""

vg = VariantGenerator()
vg.add("hidden_sizes", [(32, 32)])#, (16, 16)])
vg.add("step_size", [0.1, 0.05, 0.01])
vg.add("seed", [11, 111, 211])#, 311, 411])
vg.add("discount", [0.99, 0.999])#, 0.9999])
vg.add("gae_lambda", [1., 0.99, 0.97])#, 0.95, 0.9])#0.99, 0.999, 0.9999])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:

    env = DirectionalSwimmerEnv(turn_interval=100)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=v["hidden_sizes"]
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        discount=v["discount"],
        gae_lambda=v["gae_lambda"],
        step_size=v["step_size"],
        batch_size=50000,
        max_path_length=500,
    )

    run_experiment_lite(
        algo.train(),
        n_parallel=3,
        exp_prefix="hrl_swimmer_2",
        seed=v["seed"],
        snapshot_mode="last",
        mode="local"
    )

    # sys.exit()
