import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from sandbox.dave.rllab.regressors.gaussian_conv_mlp_regressor import GaussianConvMLPRegressor


class GaussianConvMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            regressor_args=None,
            optimizer_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianConvMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()
        self._subsample_factor = subsample_factor
        self._regressor = GaussianConvMLPRegressor(
            input_shape=env_spec.observation_space.shape,
            output_dim=1,
            name="vf",
            optimizer_args=optimizer_args,
            **regressor_args
        )

    @overrides
    def fit(self, paths):
        # --
        # Subsample before fitting.
        if self._subsample_factor < 1:
            lst_rnd_idx = []
            for path in paths:
                # Subsample index
                path_len = len(path['returns'])
                rnd_idx = np.random.choice(path_len, int(np.ceil(path_len * self._subsample_factor)),
                                           replace=False)
                lst_rnd_idx.append(rnd_idx)
            observations = np.concatenate([p["observations"][idx] for p, idx in zip(paths, lst_rnd_idx)])
            returns = np.concatenate([p["returns"][idx] for p, idx in zip(paths, lst_rnd_idx)])
        else:
            observations = np.concatenate([p["observations"] for p in paths])
            returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def fit_by_samples_data(self, samples_data):
        observations = samples_data["observations"]
        returns = samples_data["returns"]
        self._regressor.fit(observations, returns.reshape((-1, 1)))


    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
