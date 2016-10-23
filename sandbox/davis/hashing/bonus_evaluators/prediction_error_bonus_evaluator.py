import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.davis.parallel.regressors.gaussian_mlp_regressor import ParallelGaussianMLPRegressor


class PredictionErrorBonusEvaluator(Serializable):
    def __init__(
            self,
            env_spec,
            regressor=None,
            regressor_args=None,
            parallel=False):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.parallel = parallel
        if regressor_args is None:
            regressor_args = dict()
        regressor_args["input_shape"] = (env_spec.observation_space.flat_dim
                                         + env_spec.action_space.flat_dim,)
        regressor_args["output_dim"] = env_spec.observation_space.flat_dim
        if regressor is not None:
            self.regressor = regressor
        elif parallel:
            if "optimizer" not in regressor_args:
                regressor_args["optimizer"] = None
            self.regressor = ParallelGaussianMLPRegressor(**regressor_args)
        else:
            regressor_args["use_trust_region"] = False
            self.regressor = GaussianMLPRegressor(**regressor_args)

        self.errors = np.array([])
        self.normalized_errors = np.array([])
        self.itr = 0

    def fit_before_process_samples(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        state_action_pairs = np.concatenate([observations, actions], axis=1)[:-1]
        successor_states = observations[1:]
        self.regressor.fit(state_action_pairs, successor_states)
        self.itr += 1
        logger.record_tabular_misc_stat("PredictionError", self.errors)
        logger.record_tabular_misc_stat("NormalizedPredictionError", self.normalized_errors)
        self.errors = np.array([])
        self.normalized_errors = np.array([])

    def predict(self, path):
        state_action_pairs = np.concatenate([path["observations"], path["actions"]], axis=1)
        regressor_predictions = self.regressor.predict(state_action_pairs)
        error = np.linalg.norm(regressor_predictions[:-1] - path["observations"][1:], axis=1)**2
        error = np.concatenate([error, [0]], axis=0)  # To line up dimension, bonus for final is 0
        normalized_error = error / np.max(error)
        self.errors = np.concatenate([self.errors, error[:-1]])
        self.normalized_errors = np.concatenate([self.normalized_errors, normalized_error[:-1]])
        return normalized_error / (self.itr if self.itr else 1)

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        pass

    def init_rank(self, rank):
        assert self.parallel
        self.rank = rank
        self.regressor.init_rank(rank)

    def init_par_objs(self, n_parallel):
        assert self.parallel
        self.rank = None
        self.regressor.init_par_objs(n_parallel)
