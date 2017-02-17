import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.baselines.base import Baseline
from sandbox.rocky.tf.regressors.gaussian_mlp_regressor import GaussianMLPRegressor


class GaussianMLPBaseline(Baseline, Parameterized, Serializable):
    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            use_features=True,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._use_features = use_features
        obs_dim = env_spec.observation_space.flat_dim
        if use_features:
            input_dim = obs_dim * 2 + 4
        else:
            input_dim = obs_dim

        self._regressor = GaussianMLPRegressor(
            input_shape=(input_dim,),
            output_dim=1,
            name="vf",
            **regressor_args
        )

    def _features(self, path):
        obs = np.concatenate([path["observations"], [path["last_observation"]]], axis=0)
        if self._use_features:
            o = np.clip(obs, -10, 10)
            l = len(path["rewards"]) + 1
            al = (path["start_t"] + np.arange(l)).reshape(-1, 1) / 100.0
            return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)
        else:
            return obs

    @overrides
    def fit(self, paths):
        features = np.concatenate([self._features(p) for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(features, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        return self._regressor.predict(self._features(path)).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def get_params_internal(self, **tags):
        return self._regressor.get_params_internal(**tags)
