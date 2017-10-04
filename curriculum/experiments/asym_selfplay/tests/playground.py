import numpy as np

from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from curriculum.experiments.asym_selfplay.envs.alice_env import AliceEnv


def update_rewards(agent1_paths, agent2_paths, gamma):
    assert len(agent1_paths)==len(agent2_paths), 'Error, both agents need an equal number of paths.'

    for a1path,a2path in zip(agent1_paths,agent2_paths):
        t_a1 = a1path['rewards'].shape[0]
        t_a2 = a2path['rewards'].shape[1]
        a1path['rewards'] = np.zeros_like(a1path['rewards'])
        a2path['rewards'] = np.zeros_like(a2path['rewards'])
        a1path['rewards'][-1] = gamma * np.max([0, t_a2-t_a1])
        a2path['rewards'][-1] = -gamma * t_a2

    return agent1_paths, agent2_paths

regular_env = PointMazeEnv()
stop_action_env = AliceEnv(PointMazeEnv())

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
regular_paths = rollout(regular_env, regular_policy, max_path_length=100,
            animated=True, speedup=5)

wrapped_paths = rollout(stop_action_env, wrapped_policy, max_path_length=100,
            animated=True, speedup=5)

#TODO next step is the update rewards function such that agent1 and 2 only get rewards based on their own performance
# see update rewards in discriminator code
regular_paths, wrapped_paths = update_rewards(regular_paths,wrapped_paths)


print('Done')