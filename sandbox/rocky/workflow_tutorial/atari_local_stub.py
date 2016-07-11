from rllab.algos.trpo import TRPO
from sandbox.rocky.hrl.envs.atari import AtariEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = AtariEnv(game="seaquest")
policy = CategoricalMLPPolicy(env_spec=env.spec)
baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    max_path_length=4500,
    batch_size=50000,
    discount=0.99,
    gae_lambda=0.99,
)
run_experiment_lite(
    algo.train(),
    exp_prefix="atari_tutorial",
    n_parallel=4,
    snapshot_mode="last",
)
