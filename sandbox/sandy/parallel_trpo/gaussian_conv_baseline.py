
import numpy as np
import multiprocessing as mp

from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable

from sandbox.adam.parallel.util import SimpleContainer
from sandbox.sandy.parallel_trpo.gaussian_conv_regressor import ParallelGaussianConvRegressor


class ParallelGaussianConvBaseline(GaussianConvBaseline, Serializable):

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            regressor_args=None,
        ):
        Serializable.quick_init(self, locals())
        if regressor_args is None:
            regressor_args = dict()

        self._subsample_factor = subsample_factor
        self._regressor = ParallelGaussianConvRegressor(
            name="vf",
            input_shape=env_spec.observation_space.shape,
            output_dim=1,
            **regressor_args
        )

    #def __getstate__(self):
    #    """ Do not try to serialize parallel objects."""
    #    return {k: v for k, v in iter(self.__dict__.items()) if k != "_par_objs"}

    #def __setstate__(self,d):
    #    self.__dict__.update(d)

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

    @overrides
    def predict(self, path):
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
