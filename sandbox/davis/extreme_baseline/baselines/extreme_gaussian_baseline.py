import numpy as np
from scipy.linalg import hankel

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer


class ExtremeGaussianBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            lookahead=1e10,
            max_opt_itr=1000,
            use_env_noise=False,
            max_path_length=None,
            regressor_args=None,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super(ExtremeGaussianBaseline, self).__init__(env_spec, **kwargs)
        if regressor_args is None:
            regressor_args = dict()

        self.env_spec = env_spec
        self.lookahead = lookahead
        self.max_path_length = max_path_length
        self.max_opt_itr = max_opt_itr
        self.use_env_noise = use_env_noise
        self.regressor_args = regressor_args
        self._regressor = None

    def initialize_regressor(self, path):
        path_length, obs_dim = path['observations'].shape
        _, action_dim = path['actions'].shape

        if self.max_path_length is None:
            self.max_path_length = path_length

        input_size = obs_dim + action_dim * max(self.lookahead - 1, 0) + 1
        if self.use_env_noise:
            _, env_noise_dim = path['env_infos']['noise'].shape
            input_size += env_noise_dim * max(self.lookahead - 1, 0)

        # import pdb; pdb.set_trace()
        self._regressor = GaussianMLPRegressor(
            input_shape=(input_size,),
            output_dim=1,
            name="vf",
            use_trust_region=False,
            optimizer=LbfgsOptimizer(max_opt_itr=self.max_opt_itr),
            **self.regressor_args
        )

    def _features(self, path):
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
            timesteps,
            action_noise,
            env_noise,
            ], axis=1)
        return features

    @overrides
    def fit_model(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        self._regressor.fit(featmat, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path, policy):
        if self._regressor is None:
            self.initialize_regressor(path)

        return self._regressor.predict(self._features(path)).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def _lookahead_noise_matrix(self, noise):
        path_length, action_dim = noise.shape
        lookahead = min(self.lookahead, path_length - 1)
        noise = np.append(noise[1:].flatten(), [0])  # Never need first noise variable
        return hankel(noise)[::action_dim, :lookahead*action_dim]
