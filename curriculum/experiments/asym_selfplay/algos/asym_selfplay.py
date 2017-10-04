import numpy as np

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.maze.point_maze_env import PointMazeEnv
from rllab.misc import logger
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.sampler.utils import rollout
from curriculum.envs.base import UniformListStateGenerator, FixedStateGenerator
from curriculum.experiments.asym_selfplay.envs.alice_env import AliceEnv
from curriculum.logging import ExperimentLogger


class AsymSelfplay(object):

    def __init__(self, algo_alice, algo_bob, env_alice, env_bob, policy_alice, policy_bob, start_states, log_dir,
                 num_rollouts=10, gamma = 0.1, alice_factor = 0.5, alice_bonus=10):
        self.algo_alice = algo_alice
        self.algo_bob = algo_bob
        self.env_alice = env_alice
        self.env_bob = env_bob
        self.policy_alice = policy_alice
        self.policy_bob = policy_bob

        self.max_path_length = algo_alice.max_path_length
        self.num_rollouts = num_rollouts
        self.gamma = gamma
        self.optimize_alice = True
        self.optimize_bob = False
        self.start_states = start_states
        self.alice_factor = alice_factor
        self.log_dir = log_dir
        self.alice_bonus = alice_bonus

    def update_rewards(self, paths_alice, paths_bob, gamma):
        assert len(paths_alice) == len(paths_bob), 'Error, both agents need an equal number of paths.'

        for path_alice, path_bob in zip(paths_alice, paths_bob):
            t_alice = path_alice['rewards'].shape[0]
            t_bob = path_bob['rewards'].shape[0]
            path_alice['rewards'] = np.zeros_like(path_alice['rewards'])
            path_bob['rewards'] = np.zeros_like(path_bob['rewards'])
            # path_alice['rewards'][-1] = gamma * np.max([0, t_bob + alice_bonus - t_alice])
            path_alice['rewards'][-1] = gamma * max(0, self.alice_bonus + t_bob - self.alice_factor * t_alice)
            path_bob['rewards'][-1] = -gamma * t_bob

        return paths_alice, paths_bob



    def optimize_batch(self):

        # get paths
        n_starts = len(self.start_states)
        logger.log("N starts: " + str(n_starts))
        all_alice_paths = []
        self.algo_alice.current_itr = 0

        with ExperimentLogger(self.log_dir, 'last_alice', snapshot_mode='last', hold_outter_log=True):
            logger.log("Training Alice")

            for i in range(n_starts):
                self.env_alice.update_start_generator(FixedStateGenerator(self.start_states[i % n_starts]))
                logger.log("Num itrs: " + str(self.algo_alice.n_itr))
                alice_paths = self.algo_alice.train()
                all_alice_paths.extend(alice_paths)

        logger.log("All alice paths: " + str(len(all_alice_paths)))

        new_paths = [path for paths in all_alice_paths for path in paths]
        logger.log("New paths: " + str(len(new_paths)))
        new_start_states = [self.env_alice._obs2start_transform(path['observations'][-1]) for path in new_paths]
        t_alices = [path['rewards'].shape[0] for path in new_paths]
        logger.log("new start states: " + str(len(new_start_states)))

        logger.log("self.num_rollouts: " + str(self.num_rollouts))

        new_start_states = np.array(new_start_states)
        if len(new_start_states) < self.num_rollouts:
            sampled_starts = new_start_states
        else:
            sampled_starts = new_start_states[np.random.choice(np.shape(new_start_states)[0], size=self.num_rollouts)]
            #return np.array(new_start_states[np.random.choice(new_start_states.shape[0], size=self.num_rollouts)])

        return (sampled_starts, t_alices)

    def optimize(self, iter=0):

        # get paths
        n_starts = len(self.start_states)

        for itr in range(self.algo_alice.n_itr):

            paths_alice = []
            paths_bob = []
            new_start_states = []

            for i in range(self.num_rollouts):
                self.env_alice.update_start_generator(FixedStateGenerator(self.start_states[i % n_starts]))

                paths_alice.append(rollout(self.env_alice, self.policy_alice, max_path_length=self.max_path_length,
                                           animated=False))

                alice_end_obs = paths_alice[i]['observations'][-1]
                new_start_state = self.env_alice._obs2start_transform(alice_end_obs)
                new_start_states.append(new_start_state)

                self.env_bob.update_start_generator(FixedStateGenerator(new_start_state))
                paths_bob.append(rollout(self.env_bob, self.policy_bob, max_path_length=self.max_path_length,
                                         animated=False))

            # update rewards
            paths_alice, paths_bob = self.update_rewards(paths_alice=paths_alice, paths_bob=paths_bob, gamma=self.gamma)

            # optimize policies
            if self.optimize_alice:
                self.algo_alice.start_worker()
                self.algo_alice.init_opt()
                training_samples_alice = self.algo_alice.sampler.process_samples(itr=iter, paths=paths_alice)
                self.algo_alice.optimize_policy(itr=iter, samples_data=training_samples_alice)

            if self.optimize_bob:
                self.algo_bob.start_worker()
                self.algo_bob.init_opt()
                training_samples_bob = self.algo_bob.sampler.process_samples(itr=iter, paths=paths_bob)
                self.algo_bob.optimize_policy(itr=iter, samples_data=training_samples_bob)

        return np.array(new_start_states)


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
    env_a2 = AliceEnv(PointMazeEnv())

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

    asym_selfplay = AsymSelfplay(algo_alice=algo_a1, algo_bob=algo_a2, num_rollouts=num_rollouts, gamma = gamma)

    for i in range(iterations):
        asym_selfplay.optimize(i)
    print('Done')