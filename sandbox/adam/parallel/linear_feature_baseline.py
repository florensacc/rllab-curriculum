
import numpy as np
import multiprocessing as mp

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.overrides import overrides

from sandbox.adam.parallel.util import SimpleContainer


class ParallelLinearFeatureBaseline(LinearFeatureBaseline):

    def __init__(self, env_spec, reg_coeff=1e-5, low_mem=False):
        self._vec_dim = 2 * int(env_spec.observation_space.flat_dim) + 3 + 1
        self._low_mem = low_mem
        super().__init__(env_spec, reg_coeff)

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
        par_data = SimpleContainer(rank=None)
        shareds = SimpleContainer(
            feat_mat=np.reshape(
                np.frombuffer(mp.RawArray('d', self._vec_dim ** 2)),
                (self._vec_dim, self._vec_dim)),
            target_vec=np.frombuffer(mp.RawArray('d', self._vec_dim)),
        )
        mgr_objs = SimpleContainer(
            lock=mp.Lock(),
            barriers_fit=[mp.Barrier(n_parallel) for _ in range(2)],
        )
        self._par_objs = (par_data, shareds, mgr_objs)

    def set_rank(self, rank):
        par_data, _, _ = self._par_objs
        par_data.rank = rank

    @overrides
    def fit(self, paths):
        """
        Parallelized.
        """
        par_data, shareds, mgr_objs = self._par_objs

        if self._low_mem:
            feat_mat, target_vec = self._features_and_targets(paths)
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
            returns = np.concatenate([path["returns"] for path in paths])
            feat_mat = featmat.T.dot(featmat)
            target_vec = featmat.T.dot(returns)

        if par_data.rank == 0:
            shareds.feat_mat[:] = feat_mat
            shareds.target_vec[:] = target_vec
            mgr_objs.barriers_fit[0].wait()
        else:
            mgr_objs.barriers_fit[0].wait()
            with mgr_objs.lock:
                shareds.feat_mat += feat_mat
                shareds.target_vec += target_vec
        mgr_objs.barriers_fit[1].wait()

        reg_coeff = self._reg_coeff
        for _ in range(5):
            # NOTE: Could parallelize this loop, but fitting is usually fast.
            self._coeffs = np.linalg.lstsq(
                shareds.feat_mat + reg_coeff * np.identity(feat_mat.shape[1]),
                shareds.target_vec
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def _features_and_targets(self, paths):
        """
        Optional: sum path data (in matrix form) rather than concatenate, to
        keep memory footprint smaller.
        """
        feat_mat = np.zeros([self._vec_dim, self._vec_dim])
        target_vec = np.zeros([self._vec_dim, ])
        path_vec = np.zeros([1, self._vec_dim])
        max_len = max([len(path["rewards"]) for path in paths])
        t = np.arange(max_len).reshape(-1, 1) / 100.0
        t2 = t ** 2
        t3 = t ** 3
        for path in paths:
            obs = np.clip(path["observations"], -10, 10)
            for o, al, al2, al3, ret in zip(obs, t, t2, t3, path["returns"]):
                path_vec[:] = np.concatenate([o, o ** 2, al, al2, al3, [1.]])
                feat_mat += path_vec.T.dot(path_vec)
                target_vec += path_vec.squeeze() * ret
        return feat_mat, target_vec
