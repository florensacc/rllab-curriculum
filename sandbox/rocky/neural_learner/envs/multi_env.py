from __future__ import print_function
from __future__ import absolute_import
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
    def __init__(self, wrapped_env, n_episodes, episode_horizon):
        assert hasattr(wrapped_env, "reset_trial"), "The environment must implement #reset_trial()"
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, wrapped_env)
        self.n_episodes = n_episodes
        self.episode_horizon = episode_horizon
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
        self.last_action = np.zeros((self.wrapped_env.action_space.flat_dim,))
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

    def log_diagnostics(self, paths):
        # Log the avg reward for each episode
        all_episode_rewards = None
        for path in paths:
            episode_rewards = np.asarray(map(
                np.sum,
                np.split(
                    path['rewards'],
                    np.where(path['env_infos']['episode_done'])[0][:-1] + 1
                )
            ))
            if all_episode_rewards is None:
                all_episode_rewards = episode_rewards
            else:
                all_episode_rewards += episode_rewards
        for idx, tot_reward in enumerate(all_episode_rewards / len(paths)):
            logger.record_tabular('AverageEpisodeReturn(%d)' % (idx+1), tot_reward)

