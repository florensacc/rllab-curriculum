from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from rllab.misc import logger
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
            Discrete(2),
        )
        self._action_space = wrapped_env.action_space
        self.reset()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

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

    def log_diagnostics(self, paths):
        # Log the avg reward for each episode
        episode_rewards = []
        discount_episode_rewards = []
        for path in paths:
            rewards = path['rewards']
            splitter = np.where(path['env_infos']['episode_done'])[0][:-1] + 1
            episode_rewards.append(
                np.asarray(list(map(np.sum, np.split(rewards, splitter))))
            )
            discount_episode_rewards.append(
                np.asarray(list(map(self.discount_sum, np.split(rewards, splitter))))
            )
        for idx, tot_reward in enumerate(np.asarray(episode_rewards, dtype=np.float32).mean(axis=0)):
            logger.record_tabular('AverageEpisodeReturn(%d)' % (idx + 1), tot_reward)
        for idx, tot_reward in enumerate(np.asarray(discount_episode_rewards, dtype=np.float32).mean(axis=0)):
            logger.record_tabular('AverageDiscountEpisodeReturn(%d)' % (idx + 1), tot_reward)
