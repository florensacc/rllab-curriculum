from sandbox.young_clgan.envs.stop_action_env import StopActionEnv
from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout

regular_env = PointMazeEnv()
stop_action_env = StopActionEnv(PointMazeEnv())

print('Regular')
print(regular_env.action_space.low)
print(regular_env.action_space.high)

print('Wrapped')
print(stop_action_env.action_space.low)
print(stop_action_env.action_space.high)

regular_policy = GaussianMLPPolicy(
        env_spec=regular_env.spec,
        hidden_sizes=(64, 64),
        std_hidden_sizes=(16, 16)
)


wrapped_policy = GaussianMLPPolicy(
        env_spec=stop_action_env.spec,
        hidden_sizes=(64, 64),
        std_hidden_sizes=(16, 16)
)

_ = rollout(regular_env, regular_policy, max_path_length=100,
            animated=True, speedup=5)

_ = rollout(stop_action_env, wrapped_policy, max_path_length=100,
            animated=True, speedup=5)

#TODO next step is the update rewards function such that agent1 and 2 only get rewards based on their own performance
# see update rewards in discriminator code

print('Done')