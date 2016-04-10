from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.bimod_env import BimodEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(BimodEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=400,
    whole_paths=True,
    max_path_length=100,
    n_itr=4,
    discount=0.99,
    step_size=0.0001,
)
algo.train()