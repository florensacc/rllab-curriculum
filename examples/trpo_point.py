from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(PointEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
algo = TRPO()
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo.train(env=env, policy=policy, baseline=baseline)
