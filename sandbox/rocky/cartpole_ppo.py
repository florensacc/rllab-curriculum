from rllab.algos.ppo import PPO
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

algo = PPO()
env = CartpoleEnv()
baseline = LinearFeatureBaseline(env_spec=env.spec)
policy = GaussianMLPPolicy(env_spec=env.spec)

algo.train(env=env, policy=policy, baseline=baseline)

