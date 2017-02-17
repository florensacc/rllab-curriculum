from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

env = MultiEnv(wrapped_env=MABEnv(n_arms=5), n_episodes=10, episode_horizon=1, discount=0.99)
