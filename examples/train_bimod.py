from rllab.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.bimod_env import BimodEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(BimodEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = PPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=400,
    whole_paths=True,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()