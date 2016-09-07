

from rllab.envs.proxy_env import ProxyEnv
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.spaces.product import Product
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
import numpy as np


class DiscretizedEnv(ProxyEnv, Serializable):
    def __init__(self, wrapped_env, n_bins):
        Serializable.quick_init(self, locals())
        """
        :type wrapped_env: Env
        """

        original_observation_space = wrapped_env.observation_space
        original_action_space = wrapped_env.action_space

        if isinstance(original_action_space, Discrete):
            observation_space = original_observation_space
            action_space = original_action_space
        elif isinstance(original_action_space, Box):
            action_dim = original_action_space.flat_dim
            if action_dim > 1:
                observation_space = Product(
                    [original_observation_space] + [Discrete(n_bins + 1)] * (action_dim - 1)
                )
            else:
                observation_space = original_observation_space
            action_space = Discrete(n_bins)
        else:
            raise NotImplementedError

        self._observation_space = observation_space
        self._action_space = action_space
        self.action_map = self.compute_discretize_map(original_action_space, n_bins)
        self.original_observation_space = original_observation_space
        self.original_action_space = original_action_space
        self.partial_action = []
        self.last_obs = None
        self.n_bins = n_bins

        ProxyEnv.__init__(self, wrapped_env)

    def compute_discretize_map(self, action_space, n_bins):
        lows, highs = action_space.bounds
        action_map = np.zeros((action_space.flat_dim, n_bins))
        for idx, (low, high) in enumerate(zip(lows, highs)):
            interped = np.interp(np.arange(n_bins), [0, n_bins - 1], [low, high])
            action_map[idx] = interped
        return action_map

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def _get_obs(self):
        if isinstance(self.original_action_space, Discrete):
            return self.last_obs
        else:
            if self.original_action_dim <= 1:
                return self.last_obs
            return (self.last_obs,) + tuple(self.partial_action) + (self.n_bins,) * \
                (self.original_action_dim - 1 - len(self.partial_action))

    def reset(self):
        self.partial_action = []
        obs = self.wrapped_env.reset()
        self.last_obs = obs
        return self._get_obs()

    @property
    def original_action_dim(self):
        return self.original_action_space.flat_dim

    def step(self, action):
        if isinstance(self.original_action_space, Discrete):
            return self.wrapped_env.step(action)
        if self.original_action_dim <= 1:
            actual_action = self.action_map[0, action]
            return self.wrapped_env.step(actual_action)
        self.partial_action.append(action)
        if len(self.partial_action) == self.original_action_space.flat_dim:
            actual_action = self.action_map[np.arange(self.original_action_dim), self.partial_action]
            next_obs, reward, done, info = self.wrapped_env.step(actual_action)
            self.last_obs = next_obs
            self.partial_action = []
            return Step(observation=self._get_obs(), reward=reward, done=done, **info)
        return Step(observation=self._get_obs(), reward=0, done=False)

    def log_diagnostics(self, paths):
        # transform the paths to be logged by the original environment
        original_paths = []
        for path in paths:
            original_paths.append(dict(
                path,
                observations=path["observations"][:, :self.original_observation_space.flat_dim][
                             ::self.original_action_dim],
            ))
        self.wrapped_env.log_diagnostics(original_paths)
