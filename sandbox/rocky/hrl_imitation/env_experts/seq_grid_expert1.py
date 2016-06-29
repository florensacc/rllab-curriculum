from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.hrl_imitation.envs.image_grid_world import ImageGridWorld
from sandbox.rocky.hrl_imitation.fixed_clock_policy import FixedClockPolicy
from sandbox.rocky.hrl.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.hrl_imitation.envs.dummy_vec_env import DummyVecEnv

from rllab.optimizers.minibatch_dataset import BatchDataset
from rllab.envs.grid_world_env import GridWorldEnv
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.misc import logger
import itertools
import numpy as np

UP = GridWorldEnv.action_from_direction("up")
DOWN = GridWorldEnv.action_from_direction("down")
LEFT = GridWorldEnv.action_from_direction("left")
RIGHT = GridWorldEnv.action_from_direction("right")


def generate_demonstration_trajectory(size, start_pos, end_pos, action_map):
    sx, sy = start_pos
    ex, ey = end_pos
    actions = []
    if sx < ex:
        actions += action_map[DOWN] * (ex - sx)
    elif sx > ex:
        actions += action_map[UP] * (sx - ex)
    if sy < ey:
        actions += action_map[RIGHT] * (ey - sy)
    elif sy > ey:
        actions += action_map[LEFT] * (sy - ey)
    # Now we execute the list of actions in the environment
    base_map = [["."] * size for _ in range(size)]
    base_map[sx][sy] = 'S'
    base_map[ex][ey] = 'G'
    wrapped_env = ImageGridWorld(desc=base_map)
    env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
    obs = env.reset()
    observations = []
    terminal = False
    reward = 0
    for act in actions:
        next_obs, reward, terminal, _ = env.step(act)
        observations.append(obs)
        obs = next_obs
    assert terminal
    assert reward == 1
    return env, dict(
        observations=np.asarray(env.observation_space.flatten_n(observations)),
        actions=np.asarray(env.action_space.flatten_n(actions))
    )


class SeqGridExpert(object):
    def __init__(self, grid_size=5, action_map=None):
        assert grid_size == 5
        self.grid_size = grid_size
        if action_map is None:
            action_map = [
                [0, 1, 1],
                [1, 3, 3],
                [2, 2, 0],
                [3, 0, 2],
            ]
        self.action_map = action_map
        self.paths = None
        self.path_infos = None
        self.dataset = None
        self.envs = None
        self.seg_obs = None
        self.seg_actions = None

        base_map = [
            "SWGW.",
            ".W.W.",
            ".W.W.",
            ".W.W.",
            "...W.",
        ]
        # base_map = [["."] * grid_size for _ in range(grid_size)]
        # base_map[0][0] = 'S'
        # base_map[-1][-1] = 'G'
        wrapped_env = ImageGridWorld(desc=base_map)
        env = TfEnv(CompoundActionSequenceEnv(wrapped_env, action_map, obs_include_history=True))
        self.template_env = env

        self.env_spec = env.spec

    def build_dataset(self, batch_size):
        if self.dataset is None:
            paths = self.generate_demonstrations()
            observations = tensor_utils.concat_tensor_list([p["observations"] for p in paths])
            actions = tensor_utils.concat_tensor_list([p["actions"] for p in paths])
            N = observations.shape[0]
            seg_obs = observations.reshape((N / 3, 3, -1))
            seg_actions = actions.reshape((N / 3, 3, -1))
            self.dataset = BatchDataset([seg_obs, seg_actions], batch_size=batch_size)
            self.seg_obs = seg_obs
            self.seg_actions = seg_actions
        return self.dataset

    def generate_demonstrations(self):
        if self.paths is None:
            logger.log("generating paths")
            all_pos = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
            paths = []
            envs = []
            for start_pos, end_pos in itertools.permutations(all_pos, 2):
                # generate a demonstration trajectory from the start to the end
                env, path = generate_demonstration_trajectory(self.grid_size, start_pos, end_pos, self.action_map)
                paths.append(path)
                envs.append(env)
            logger.log("generated")
            # import ipdb; ipdb.set_trace()
            self.paths = paths
            self.envs = envs
        return self.paths

    def log_diagnostics(self, algo):
        logger.log("logging MI...")
        self.log_mis(algo)
        # logger.log("logging train stats...")
        # self.log_train_stats(algo)
        logger.log("logging test stats...")
        self.log_test_stats(algo)

    def log_mis(self, algo):

        low_policy = algo.low_policy

        observations = self.seg_obs[:, 0, :]
        actions = self.seg_actions[:, 0, :]
        N = observations.shape[0]

        all_low_probs = []

        for g in xrange(algo.subgoal_dim):
            subgoals = np.tile(
                algo.high_policy.action_space.flatten(g).reshape((1, -1)),
                (N, 1)
            )
            low_obs = np.concatenate([observations, subgoals], axis=-1)
            low_probs = algo.low_policy.dist_info(low_obs)["prob"]
            all_low_probs.append(low_probs)

        all_low_probs = np.asarray(all_low_probs)
        flat_low_probs = all_low_probs.reshape((-1, algo.low_policy.action_space.n))

        p_a_given_s = np.mean(all_low_probs, axis=0)
        h_a_given_s = np.mean(algo.low_policy.distribution.entropy(dict(prob=p_a_given_s)))
        h_a_given_h_s = np.mean(algo.low_policy.distribution.entropy(dict(prob=flat_low_probs)))

        mi_a_h_given_s = h_a_given_s - h_a_given_h_s

        logger.record_tabular("I(a;h|s)", mi_a_h_given_s)
        logger.record_tabular("H(a|s)", h_a_given_s)
        logger.record_tabular("H(a|h,s)", h_a_given_h_s)

    def log_train_stats(self, algo):
        env_spec = self.env_spec
        trained_policy = FixedClockPolicy(env_spec=env_spec, high_policy=algo.high_policy, low_policy=algo.low_policy,
                                          subgoal_interval=algo.subgoal_interval)

        n_envs = len(self.envs)

        train_venv = DummyVecEnv(env=self.template_env, n=n_envs, envs=self.envs,
                                 max_path_length=algo.max_path_length)

        path_rewards = [None] * n_envs
        path_discount_rewards = [None] * n_envs
        obses = train_venv.reset()
        dones = np.asarray([True] * n_envs)
        for t in xrange(algo.max_path_length):
            trained_policy.reset(dones)
            acts, _ = trained_policy.get_actions(obses)
            next_obses, rewards, dones, _ = train_venv.step(acts)
            obses = next_obses
            for idx, done in enumerate(dones):
                if done and path_rewards[idx] is None:
                    path_rewards[idx] = rewards[idx]
                    path_discount_rewards[idx] = rewards[idx] * (algo.discount ** t)

        logger.record_tabular("AverageTrainReturn", np.mean(path_rewards))
        logger.record_tabular("AverageTrainDiscountedReturn", np.mean(path_discount_rewards))

    def log_test_stats(self, algo):
        env_spec = self.env_spec
        test_policy = FixedClockPolicy(env_spec=env_spec, high_policy=algo.alt_high_policy, low_policy=algo.low_policy,
                                       subgoal_interval=algo.subgoal_interval)

        n_envs = 100

        test_venv = DummyVecEnv(env=self.template_env, n=n_envs, max_path_length=algo.max_path_length)

        path_rewards = [None] * n_envs
        path_discount_rewards = [None] * n_envs
        obses = test_venv.reset()
        dones = np.asarray([True] * n_envs)
        for t in xrange(algo.max_path_length):
            test_policy.reset(dones)
            acts, _ = test_policy.get_actions(obses)
            next_obses, rewards, dones, _ = test_venv.step(acts)
            obses = next_obses
            for idx, done in enumerate(dones):
                if done and path_rewards[idx] is None:
                    path_rewards[idx] = rewards[idx]
                    path_discount_rewards[idx] = rewards[idx] * (algo.discount ** t)

        logger.record_tabular("AverageTestReturn", np.mean(path_rewards))
        logger.record_tabular("AverageTestDiscountedReturn", np.mean(path_discount_rewards))
