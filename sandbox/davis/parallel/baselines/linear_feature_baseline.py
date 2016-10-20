
import numpy as np
import multiprocessing as mp

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.overrides import overrides

from sandbox.adam.parallel.util import SimpleContainer


class ParallelLinearFeatureBaseline(LinearFeatureBaseline):

    def __init__(self, env_spec, reg_coeff=1e-5, low_mem=False):
        self._vec_dim = 2 * int(env_spec.observation_space.flat_dim) + 3 + 1
        self._low_mem = low_mem
        if low_mem:
            self.feat_mat = np.zeros([self._vec_dim, self._vec_dim])
            self.target_vec = np.zeros([self._vec_dim, ])
            self.path_vec = np.zeros([1, self._vec_dim])
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
        self.rank = None
        shareds = SimpleContainer(
            feat_mat=np.reshape(
                np.frombuffer(mp.RawArray('d', (self._vec_dim ** 2) * n_parallel)),
                (self._vec_dim, self._vec_dim, n_parallel)),
            target_vec=np.reshape(
                np.frombuffer(mp.RawArray('d', self._vec_dim * n_parallel)),
                (self._vec_dim, n_parallel)),
            coeffs=np.frombuffer(mp.RawArray('d', self._vec_dim)),
        )
        barriers = SimpleContainer(
            fit=[mp.Barrier(n_parallel) for _ in range(2)],
        )
        self._coeffs = shareds.coeffs  # (compatible with inherited predict() method)
        self._par_objs = (shareds, barriers)

    def init_rank(self, rank):
        self.rank = rank

    @overrides
    def fit(self, paths):
        """
        Parallelized.
        """
        shareds, barriers = self._par_objs

        if self._low_mem:
            self._features_and_targets(paths)
            f_mat = self.feat_mat
            t_vec = self.target_vec
        else:
            featmat = np.concatenate([self._features(path) for path in paths])
            returns = np.concatenate([path["returns"] for path in paths])
            f_mat = featmat.T.dot(featmat)
            t_vec = featmat.T.dot(returns)

        shareds.feat_mat[:, :, self.rank] = f_mat
        shareds.target_vec[:, self.rank] = t_vec
        barriers.fit[0].wait()
        if self.rank == 0:
            feat_mat = np.sum(shareds.feat_mat, axis=2)
            target_vec = np.squeeze(np.sum(shareds.target_vec, axis=1))

            reg_coeff = self._reg_coeff
            for _ in range(5):
                # NOTE: Could parallelize this loop, but fitting is usually fast.
                shareds.coeffs[:] = np.linalg.lstsq(
                    feat_mat + reg_coeff * np.identity(feat_mat.shape[1]),
                    target_vec
                )[0]
                if not np.any(np.isnan(shareds.coeffs)):
                    break
                reg_coeff *= 10
        barriers.fit[1].wait()

    def _features_and_targets(self, paths):
        """
        Optional: sum path data (in matrix form) rather than concatenate, to
        keep memory footprint smaller.
        """
        self.feat_mat.fill(0.)
        self.target_vec.fill(0.)
        self.path_vec.fill(0.)
        max_len = max([len(path["rewards"]) for path in paths])
        t = np.arange(max_len).reshape(-1, 1) / 100.0
        t2 = t ** 2
        t3 = t ** 3
        for path in paths:
            obs = np.clip(path["observations"], -10, 10)
            for o, al, al2, al3, ret in zip(obs, t, t2, t3, path["returns"]):
                self.path_vec[:] = np.concatenate([o, o ** 2, al, al2, al3, [1.]])
                self.feat_mat += self.path_vec.T.dot(self.path_vec)
                self.target_vec += self.path_vec.squeeze() * ret

    def force_compile(self):
        pass
