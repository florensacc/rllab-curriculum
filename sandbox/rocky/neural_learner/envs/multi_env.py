from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from rllab.misc import logger as root_logger
import numpy as np

BIG = 50000


class MultiEnv(ProxyEnv, Serializable):
    def __init__(self, wrapped_env, n_episodes, episode_horizon, discount):
        assert hasattr(wrapped_env, "reset_trial"), "The environment must implement #reset_trial()"
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, wrapped_env)
        self.n_episodes = n_episodes
        self.episode_horizon = episode_horizon
        self.discount = discount
        self.last_action = None
        self.last_reward = None
        self.last_terminal = None
        self.cnt_episodes = None
        self.episode_t = None
        self._observation_space = Product(
            wrapped_env.observation_space,
            wrapped_env.action_space,
            Box(low=-BIG, high=BIG, shape=(1,)),
            Box(low=0, high=1, shape=(1,)),
        )
        self._action_space = wrapped_env.action_space
        self.reset()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def vectorized(self):
        return getattr(self.wrapped_env, 'vectorized', False)

    def vec_env_executor(self, n_envs):
        return VecMultiEnv(n_envs=n_envs, env=self)

    def reset(self):
        self.last_reward = 0
        self.last_action = self.wrapped_env.action_space.default_value
        self.last_terminal = 1
        self.cnt_episodes = 0
        self.episode_t = 0
        return self.convert_obs(self.wrapped_env.reset_trial())

    def convert_obs(self, obs):
        return obs, self.last_action, self.last_reward, self.last_terminal

    def step(self, action):
        next_obs, reward, done, info = self.wrapped_env.step(action)
        self.last_reward = reward
        self.last_action = action
        self.last_terminal = int(done)
        self.episode_t += 1

        if done or self.episode_t >= self.episode_horizon:
            self.cnt_episodes += 1
            self.episode_t = 0
            self.last_terminal = 1
            next_obs = self.wrapped_env.reset()

        trial_done = self.cnt_episodes >= self.n_episodes

        return Step(self.convert_obs(next_obs), reward, trial_done, **dict(info, episode_done=self.last_terminal))

    def discount_sum(self, rewards):
        return np.sum(self.discount ** np.arange(len(rewards)) * rewards)

    def log_diagnostics(self, paths, *args, **kwargs):  # logger=None):
        if 'logger' in kwargs:
            logger = kwargs['logger']
        else:
            logger = root_logger
        # Log the avg reward for each episode
        episode_rewards = [[] for _ in range(self.n_episodes)]
        discount_episode_rewards = [[] for _ in range(self.n_episodes)]
        episode_success = [[] for _ in range(self.n_episodes)]
        episode_lens = [[] for _ in range(self.n_episodes)]
        for path in paths:
            rewards = path['rewards']
            splitter = np.where(path['env_infos']['episode_done'])[0][:-1] + 1
            split_rewards = np.split(rewards, splitter)
            if 'success' in path['env_infos']:
                split_success = [x[-1] for x in np.split(path['env_infos']['success'], splitter)]
            else:
                split_success = np.zeros((len(split_rewards),))
            # episode_rewards.append(
            #     np.asarray(list(map(np.sum, split_rewards)))
            # )
            # discount_episode_rewards.append(
            #     np.asarray(list(map(self.discount_sum, split_rewards)))
            # )
            for epi, (rews, success) in enumerate(zip(split_rewards, split_success)):
                if success:
                    episode_lens[epi].append(len(rews))
                episode_success[epi].append(success)
                episode_rewards[epi].append(np.sum(rews))
                discount_episode_rewards[epi].append(self.discount_sum(rews))

        def log_stat(name, data):
            avg_data = list(map(np.mean, data))
            # for idx, entry in enumerate(avg_data):
            #     logger.record_tabular('Average%s(%d)' % (name, idx + 1), entry)
            # for idx, (entry, next_entry) in enumerate(zip(avg_data, avg_data[1:])):
            #     logger.record_tabular('Delta%s(%d)' % (name, idx + 1), next_entry - entry)
            logger.record_tabular('Average%s(First)' % name, avg_data[0], op=np.mean)
            logger.record_tabular('Average%s(Last)' % name, avg_data[-1], op=np.mean)
            logger.record_tabular('Delta%s(Last-First)' % name, avg_data[-1] - avg_data[0], op=np.mean)

        log_stat('EpisodeReturn', episode_rewards)
        log_stat('DiscountEpisodeReturn', discount_episode_rewards)
        if 'success' in paths[0]['env_infos']:
            log_stat('SuccessEpisodeLength', episode_lens)
            log_stat('SuccessRate', episode_success)

        if hasattr(self.wrapped_env, 'log_diagnostics_multi'):
            self.wrapped_env.log_diagnostics_multi(multi_env=self, paths=paths, *args, **kwargs)
            # for idx, tot_reward in enumerate(map(np.mean, episode_rewards)):
            #     logger.record_tabular('AverageEpisodeReturn(%d)' % (idx + 1), tot_reward)
            # for idx, tot_reward in enumerate(map(np.mean, episode_rewards)):
            #     logger.record_tabular('AverageEpisodeReturn(%d)' % (idx + 1), tot_reward)
            # for idx, tot_reward in enumerate(map(np.mean, discount_episode_rewards)):
            #     logger.record_tabular('AverageDiscountEpisodeReturn(%d)' % (idx + 1), tot_reward)
            # for idx, lens in enumerate(map(np.mean, episode_lens)):
            #     logger.record_tabular('AverageSuccessEpisodeLength(%d)' % (idx + 1), lens)
            # for idx, success in enumerate(map(np.mean, episode_success)):
            #     logger.record_tabular('AverageSuccessRate(%d)' % (idx + 1), success)


class VecMultiEnv(object):
    def __init__(self, n_envs, env):
        assert getattr(env, 'vectorized', False)
        self.n_envs = n_envs
        self.env = env
        self.vec_env = env.wrapped_env.vec_env_executor(n_envs=n_envs)
        self.last_rewards = np.zeros((self.n_envs,))
        self.last_actions = np.zeros((self.n_envs,) + np.shape(env.wrapped_env.action_space.default_value))
        self.last_terminals = np.zeros((self.n_envs,))
        self.cnt_episodes = np.zeros((self.n_envs,), dtype=np.int)
        self.trial_seeds = np.zeros((self.n_envs,), dtype=np.int)
        self.episode_t = np.zeros((self.n_envs,))
        self.ts = np.zeros((self.n_envs,))
        self.reset()

    @property
    def num_envs(self):
        return self.n_envs

    def reset(self, dones=None):
        if dones is None:
            dones = np.asarray([True] * self.n_envs)
        else:
            dones = np.cast['bool'](dones)
        self.last_rewards[dones] = 0
        default_action = self.env.wrapped_env.action_space.default_value
        self.last_actions[dones] = default_action
        self.last_terminals[dones] = 1
        self.cnt_episodes[dones] = 0
        self.episode_t[dones] = 0
        self.ts[dones] = 0
        self.trial_seeds[dones] = np.random.randint(low=0, high=np.iinfo(np.int32).max, size=np.sum(dones))
        return self.convert_obs(self.vec_env.reset_trial(dones, seeds=self.trial_seeds[dones]))

    def step(self, actions, max_path_length):
        next_obs, rewards, dones, infos = self.vec_env.step(actions, max_path_length=None)
        self.last_rewards = np.copy(rewards)
        self.last_actions = np.copy(actions)
        self.last_terminals = np.cast['int'](dones)
        self.episode_t += 1
        self.ts += 1

        episode_ids = np.copy(self.cnt_episodes)
        trial_seeds = np.copy(self.trial_seeds)

        dones[self.episode_t >= self.env.episode_horizon] = True

        if np.any(dones):
            self.cnt_episodes[dones] += 1
            self.episode_t[dones] = 0
            self.last_terminals[dones] = 1
            next_obs[dones] = self.vec_env.reset(dones)

        trial_done = self.cnt_episodes >= self.env.n_episodes
        if max_path_length is not None:
            trial_done[self.ts >= max_path_length] = True

        next_obs = self.convert_obs(next_obs)

        if np.any(trial_done):
            reset_obs = self.reset(trial_done)
            reset_idx = 0
            for idx, done in enumerate(trial_done):
                if done:
                    next_obs[idx] = reset_obs[reset_idx]
                    reset_idx += 1

        infos = dict(
            infos,
            episode_done=np.copy(self.last_terminals),
            episode_id=episode_ids,
            trial_seed=trial_seeds,
        )
        return next_obs, rewards, trial_done, infos

    def convert_obs(self, obs):
        return list(zip(obs, self.last_actions, self.last_rewards, self.last_terminals))

    def handle_policy_reset(self, policy, dones):
        policy.reset(dones)
