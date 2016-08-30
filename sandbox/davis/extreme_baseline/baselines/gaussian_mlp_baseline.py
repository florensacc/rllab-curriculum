import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer


class GaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            max_opt_itr=1000,
            regressor_args=None,
    ):
        Serializable.quick_init(self, locals())
        super(GaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim * num_seq_inputs + 1,),
            output_dim=1,
            name="vf",
            use_trust_region=False,
            optimizer=LbfgsOptimizer(max_opt_itr=max_opt_itr),
            **regressor_args
        )

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        path_length, _ = paths[0]['observations'].shape  # For now, assumes uniform path length (e.g. Half Cheetah)
        timesteps = np.arange(path_length).reshape(-1, 1) / float(path_length)
        returns = np.concatenate([p["returns"] for p in paths])
        observations_with_timestep = np.concatenate([observations, np.repeat(timesteps, len(paths), axis=0)], axis=1)
        self._regressor.fit(observations_with_timestep, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path, policy):
        path_length, _ = path['observations'].shape
        timesteps = np.arange(path_length).reshape(-1, 1) / float(path_length)
        observations_with_timestep = np.concatenate([path['observations'], timesteps], axis=1)
        return self._regressor.predict(observations_with_timestep).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
