from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides

import numpy as np
from scipy.linalg import hankel


class ExtremeLinearBaseline(Baseline):
    def __init__(
            self,
            env_spec,
            lookahead=1e10,
            reg_coeff=1e-5,
            use_env_noise=False,
            max_path_length=None,
            **kwargs
            ):
        """
        Without the max_path_length argument, if the environment terminates rollouts early,
        everything will break.
        """
        super(ExtremeLinearBaseline, self).__init__(env_spec, **kwargs)
        self._coeffs = None
        self.reg_coeff = reg_coeff
        self.lookahead = lookahead
        self.use_env_noise = use_env_noise
        self.max_path_length = max_path_length

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        if self.max_path_length is None:
            self.max_path_length = len(path['observations'])

        observations = np.clip(path['observations'], -10, 10)
        path_length, observation_dim = observations.shape

        timesteps = np.arange(path_length).reshape(-1, 1) / 100.0

        action_noise = path['agent_infos']['noise']
        _, action_dim = action_noise.shape
        if path_length < self.max_path_length:
            action_noise = np.concatenate(
                [action_noise, np.zeros((self.max_path_length - path_length, action_dim))])
        action_noise = self._lookahead_noise_matrix(action_noise)[:path_length]

        if self.use_env_noise:
            env_noise = path['env_infos']['noise']
            _, env_noise_dim = env_noise.shape
            if path_length < self.max_path_length:
                env_noise = np.concatenate(
                    [env_noise, np.zeros((self.max_path_length - path_length, env_noise_dim))])
            env_noise = self._lookahead_noise_matrix(env_noise)[:path_length]
        else:
            env_noise = np.ndarray((path_length, 0))

        features = np.concatenate([
            observations,
            observations**2,
            timesteps,
            timesteps**2,
            timesteps**3,
            action_noise,
            action_noise**2,
            env_noise,
            env_noise**2,
            np.ones((path_length, 1))
            ], axis=1)
        return features

    def _lookahead_noise_matrix(self, noise):
        path_length, action_dim = noise.shape
        lookahead = min(self.lookahead, path_length - 1)
        noise = np.append(noise[1:].flatten(), [0])  # Never need first noise variable
        return hankel(noise)[::action_dim, :lookahead*action_dim]

    @overrides
    def fit_model(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths])
        self._coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + self.reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(returns)
        )[0]

    @overrides
    def predict(self, path, policy):
        if self._coeffs is None:
            return np.zeros(len(path['rewards']))
        return self._features(path).dot(self._coeffs)
