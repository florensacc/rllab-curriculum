import numpy as np

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from sandbox.young_clgan.experiments.asym_selfplay.envs.stop_action_env import StopActionEnv


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


class AsymSelfplay(object):

    def __init__(self, algo_a1, algo_a2, num_rollouts=10, gamma = 0.01):
        self.algo_a1 = algo_a1
        self.algo_a2 = algo_a2
        self.env_a1 = algo_a1.env
        self.env_a2 = algo_a2.env
        self.policy_a1 = algo_a1.policy
        self.policy_a2 = algo_a2.policy
        self.max_path_length = algo_a1.max_path_length
        self.num_rollouts = num_rollouts
        self.gamma = gamma

    def optimize(self, itr):

        # get paths
        a1_paths = []
        a2_paths = []
        for i in range(self.num_rollouts):
            # TODO command for resetting to goal position
            self.env_a1.reset('to the goal position')
            a1_paths.append(rollout(self.env_a1, self.policy_a1, max_path_length=self.max_path_length,
                                    animated=False))
            # todo which part of observation for reset
            self.env_a2.reset(a1_paths[i]['observations'][-1])
            a2_paths.append(rollout(self.env_a2, self.policy_a2, max_path_length=self.max_path_length,
                                    animated=False))

        # update rewards
        update_rewards(agent1_paths=a1_paths, agent2_paths=a2_paths,gamma=self.gamma)

        # extract samples
        a1_training_samples = self.algo_a1.process_samples(itr=itr, paths=a1_paths)
        a2_training_samples = self.algo_a2.process_samples(itr=itr, paths=a2_paths)

        # optimise policies
        self.algo_a1.optimize_policy(itr=itr, samples_data=a1_training_samples)
        self.algo_a2.optimize_policy(itr=itr, samples_data=a2_training_samples)


if __name__ == '__main__':
    # aym selfplay only uses a single rollout
    # batching should be more stable
    num_rollouts = 50
    iterations = 10
    max_path_length = 100
    # they use a gamma between 0.1 and 0.01
    gamma = 0.01

    # todo setup the correct environments (correct wrappers for arbitrary reset)
    env_a1 = PointMazeEnv()
    env_a2 = StopActionEnv(PointMazeEnv())

    policy_a1 = GaussianMLPPolicy(
            env_spec=env_a1.spec,
            hidden_sizes=(64, 64),
            std_hidden_sizes=(16, 16)
    )

    policy_a2 = GaussianMLPPolicy(
            env_spec=env_a2.spec,
            hidden_sizes=(64, 64),
            std_hidden_sizes=(16, 16)
    )

    baseline_a1 = LinearFeatureBaseline(env_spec=env_a1.spec)
    baseline_a2 = LinearFeatureBaseline(env_spec=env_a2.spec)


    # They use discrete policy gradient but we should compare based on the same optimiser as TRPO tends to be mroe robust
    # Their algo: R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement
    # learning. In Machine Learning, pages 229–256, 1992.

    algo_a1 = TRPO(
        env=env_a1,
        policy=policy_a1,
        baseline=baseline_a1,
        batch_size=4000,
        max_path_length=max_path_length,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        # plot=True,
    )

    algo_a2 = TRPO(
        env=env_a2,
        policy=policy_a2,
        baseline=baseline_a2,
        batch_size=4000,
        max_path_length=max_path_length,
        n_itr=40,
        discount=0.99,
        step_size=0.01,
        # plot=True,
    )

    asym_selfplay = AsymSelfplay(algo_a1=algo_a1, algo_a2=algo_a2, num_rollouts=num_rollouts, gamma = gamma)

    for i in range(iterations):
        asym_selfplay.optimize(i)
    print('Done')