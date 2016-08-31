from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.humanoid_env import HumanoidEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

env = normalize(Walker2DEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=10000,
    whole_paths=True,
    max_path_length=1000,
    n_itr=1000,
    discount=0.995,
    step_size=0.01,
    gae_lambda=0.98,
)

run_experiment_lite(
    algo.train(),
    n_parallel=1,
    snapshot_mode="all",
    exp_prefix="cs188",
)
