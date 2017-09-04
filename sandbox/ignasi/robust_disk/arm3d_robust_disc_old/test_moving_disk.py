from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from examples.point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.ignasi.envs.action_limited_env import ActionLimitedEnv
from sandbox.ignasi.envs.arm3d.arm3d_disc_robust_env import Arm3dDiscRobustEnv

inner_env = normalize(Arm3dDiscRobustEnv())
env = ActionLimitedEnv(inner_env)
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=5000,
    max_path_length=5000,
    plot=True,
)
algo.train()
