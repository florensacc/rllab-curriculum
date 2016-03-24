from rllab.algo.ppo import PPO
from rllab.env.box2d.cartpole_env import CartpoleEnv
from rllab.baseline.linear_feature_baseline import LinearFeatureBaseline
from rllab.policy.gaussian_mlp_policy import GaussianMLPPolicy

algo = PPO()
env = CartpoleEnv()
baseline = LinearFeatureBaseline(env_spec=env.spec)
policy = GaussianMLPPolicy(env_spec=env.spec)

algo.train(env=env, policy=policy, baseline=baseline)

