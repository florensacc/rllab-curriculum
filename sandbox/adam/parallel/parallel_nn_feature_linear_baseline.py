
import numpy as np
import multiprocessing as mp

from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.overrides import overrides

from sandbox.adam.parallel.util import SimpleContainer


class ParallelNNFeatureLinearBaseline(LinearFeatureBaseline):
    """
    See Adam's ParallelLinearFeatureBaseline for implementation details.
    The only change is taking features from a certain layer from a NN policy.

    parallel least squares:
    A_i: feature matrix computed by process i
    b_i: regression target computed by process i

    argmin_x \sum_i \|A_i x - b_i\|_2^2 = (\sum_i A_i^T A_i)^{-1} (\sum_i A_i^T b)

    Each process computes A_i^T A_i and A_i^T b_i. A master process computes the least squares. Notice that the size of communicated data is determined by the feature dimension, not by the number of samples.
    Alternatively, further decompose A_i^T A_i in the same way into computation over different paths. This way memory requirement is smaller. See _features_and_targets for details.
    """

    def __init__(
            self,
            env_spec,
            policy,
            nn_feature_power=2,
            t_power=3,
            reg_coeff=1e-5,
            prediction_clip=1000,
        ):
        self._policy = policy
        self._nn_feature_power = nn_feature_power
        self._t_power = t_power
        nn_feature_len = np.prod(policy.get_feature_shape())
        self._nn_feature_len = nn_feature_len
        self._vec_dim = int(nn_feature_len * nn_feature_power + t_power + 1)
        self._prediction_clip = prediction_clip
        super().__init__(env_spec, reg_coeff)

    @overrides
    def _features(self, path):
        features_list = []

        nn_features = self._policy.get_features(path["observations"])
        nn_features_flat = nn_features.reshape((nn_features.shape[0], self._nn_feature_len))
        for i in range(1, self._nn_feature_power + 1):
            features_list.append(nn_features_flat ** i)

        path_len = len(path["rewards"])
        tt = np.arange(path_len).reshape(-1, 1) / 100.0 # 100 is heuristically chosen for most problems we care about
        for i in range(0,self._t_power + 1): # also includes bias terms
            features_list.append(tt ** i)

        return np.concatenate(features_list, axis=1)

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
            # A^T A
            feat_mat=np.reshape(
                np.frombuffer(mp.RawArray('d', (self._vec_dim ** 2) * n_parallel)),
                (self._vec_dim, self._vec_dim, n_parallel)),
            # A^T b
            target_vec=np.reshape(
                np.frombuffer(mp.RawArray('d', self._vec_dim * n_parallel)),
                (self._vec_dim, n_parallel)),
            # x
            coeffs=np.frombuffer(mp.RawArray('d', self._vec_dim)),
        )
        barriers = SimpleContainer(
            fit=[mp.Barrier(n_parallel) for _ in range(2)],
        )
        self._coeffs = shareds.coeffs  # (compatible with inherited predict() method)
        self._par_objs = (shareds, barriers)

    def force_compile(self):
        pass

    def init_rank(self, rank):
        self.rank = rank

    @overrides
    def fit(self, paths):
        """
        Parallelized.
        """
        shareds, barriers = self._par_objs

        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        f_mat = featmat.T.dot(featmat)
        t_vec = featmat.T.dot(returns)

        shareds.feat_mat[:, :, self.rank] = f_mat
        shareds.target_vec[:, self.rank] = t_vec
        barriers.fit[0].wait()
        # master process computes least squares
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

            # compute training error
            train_error = np.average((featmat.dot(self._coeffs) - returns)**2)
            print("---------- baseline training error: %f"%(train_error))
        barriers.fit[1].wait()

    @overrides
    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        predictions = self._features(path).dot(self._coeffs)
        clipped_predictions = np.clip(predictions,-self._prediction_clip, self._prediction_clip)
        return clipped_predictions

    def _features_and_targets(self, paths):
        """
        Optional: sum path data (in matrix form) rather than concatenate, to
        keep memory footprint smaller.
        """
        raise NotImplementedError
        # self.feat_mat.fill(0.)
        # self.target_vec.fill(0.)
        # self.path_vec.fill(0.)
        # max_len = max([len(path["rewards"]) for path in paths])
        # t = np.arange(max_len).reshape(-1, 1) / 100.0
        # t2 = t ** 2
        # t3 = t ** 3
        # for path in paths:
        #     obs = np.clip(path["observations"], -10, 10)
        #     for o, al, al2, al3, ret in zip(obs, t, t2, t3, path["returns"]):
        #         self.path_vec[:] = np.concatenate([o, o ** 2, al, al2, al3, [1.]])
        #         self.feat_mat += self.path_vec.T.dot(self.path_vec)
        #         self.target_vec += self.path_vec.squeeze() * ret
