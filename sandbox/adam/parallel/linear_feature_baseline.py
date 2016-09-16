from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np
import multiprocessing as mp
from sandbox.adam.parallel.util import SimpleContainer


class ParallelLinearFeatureBaseline(Baseline):

    def __init__(self, env_spec, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self._vec_dim = 2 * int(env_spec.observation_space.flat_dim) + 3 + 1
        self._par_data = None  # objects for parallelism, to be populated
        self._shareds = None
        self._mgr_objs = None

    @overrides
    @property
    def algorithm_parallelized(self):
        return True

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def init_par_objs(self, n_parallel):
        """
        These objects will be inherited by forked subprocesses.
        (Also possible to return these and attach them explicitly within
        subprocess--neeeded in Windows.)
        """
        par_data = SimpleContainer(rank=None)

        shareds = SimpleContainer(
            feat_mat=np.reshape(
                np.frombuffer(mp.RawArray('d', self._vec_dim ** 2)),
                (self._vec_dim, self._vec_dim)),
            target_vec=np.reshape(
                np.frombuffer(mp.RawArray('d', self._vec_dim)),
                (self._vec_dim, 1)),
        )

        mgr_objs = SimpleContainer(
            lock=mp.Lock(),
            barriers_fit=[mp.Barrier(n_parallel) for _ in range(2)],
        )

        self._par_objs = (par_data, shareds, mgr_objs)

    def update_rank(self, rank):
        par_data, _, _ = self._par_objs
        par_data.rank = rank

    def _all_features_and_targets(self, paths):
        """
        Used in parallel implementation: sum path data (in matrix form) rather
        than concatenate, to share a small set of values for fitting.
        """
        feat_mat = np.zeros([self._vec_dim, self._vec_dim])
        target_vec = np.zeros([1, self._vec_dim])
        path_vec = np.zeros([1, self._vec_dim])
        max_len = max([len(path["rewards"]) for path in paths])
        times = np.arange(max_len) / 100.0
        times_2 = times ** 2
        times_3 = times ** 3
        for path in paths:
            obs = np.clip(path["observations"], -10, 10)
            for o, al, al_2, al_3, r in zip(obs, times, times_2, times_3, path["rewards"]):
                path_vec[:] = [o, o ** 2, al, al_2, al_3, 1]
                feat_mat += path_vec.T.dot(path_vec)
                target_vec += path_vec * r
        return feat_mat, target_vec.T

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        h = len(path["rewards"])
        al = np.arange(h).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((h, 1))], axis=1)

    @overrides
    def fit(self, paths, par_objs):
        """
        This method is parallelized.
        """
        par_data, shareds, mgr_objs = self._par_objs

        feat_mat, target_vec = self._all_features_and_targets(paths)
        if par_data.rank == 0:
            shareds.feat_mat.fill(0.)
            shareds.target_vec.fill(0.)
        mgr_objs.barriers_fit[0].wait()
        with mgr_objs.lock:
            shareds.feat_mat += feat_mat
            shareds.target_vec += target_vec
        mgr_objs.barriers_fit[1].wait()
        reg_coeff = self._reg_coeff
        for _ in range(5):
            # NOTE: Could parallelize this loop, but fittin is usually fast.
            self._coeffs = np.linalg.lstsq(
                shareds.feat_mat + reg_coeff * np.identity(feat_mat.shape[1]),
                target_vec
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    @overrides
    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)
