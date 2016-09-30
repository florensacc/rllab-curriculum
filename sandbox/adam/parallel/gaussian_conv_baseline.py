
import numpy as np
import multiprocessing as mp

from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable

from sandbox.adam.parallel.util import SimpleContainer
from sandbox.adam.parallel.gaussian_conv_regressor import ParallelGaussianConvRegressor


class ParallelGaussianConvBaseline(GaussianConvBaseline):

    def __init__(
            self,
            env_spec,
            regressor_args=None,
        ):
        Serializable.quick_init(self,locals())
        # super().__init__(env_spec)
        self._mdp_spec = env_spec # hack
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = ParallelGaussianConvRegressor(
            name="vf",
            input_shape=env_spec.observation_space.shape,
            output_dim=1,
            **regressor_args
        )

    def __getstate__(self):
        """ Do not try to serialize parallel objects."""
        return {k: v for k, v in iter(self.__dict__.items()) if k != "_par_objs"}

    @overrides
    @property
    def algorithm_parallelized(self):
        return True

    def init_par_objs(self, n_parallel):
        """
        These objects will be inherited by forked subprocesses.
        (Also possible to return these and attach them explicitly within
        subprocess--neeeded in Windows.)
        """
        self.rank = None
        self._regressor.init_par_objs(n_parallel)

    def init_rank(self, rank):
        self.rank = rank
        self._regressor.init_rank(rank)

    def force_compile(self):
        self._regressor.force_compile()

    @overrides
    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
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
