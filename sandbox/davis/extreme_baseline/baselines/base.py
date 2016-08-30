from rllab.misc import autoargs
import rllab.misc.logger as logger
import numpy as np


class Baseline(object):

    def __init__(self, env_spec, validate=False, title="Baseline"):
        self._mdp_spec = env_spec
        self.validate = validate
        self.title = title

    @property
    def algorithm_parallelized(self):
        return False

    def get_param_values(self):
        raise NotImplementedError

    def set_param_values(self, val):
        raise NotImplementedError

    def fit(self, paths, policy=None):
        if self.validate:
            self.log_train_and_validation_error(paths, policy)
        self.fit_model(paths)

    def fit_model(self, paths):
        raise NotImplementedError

    def predict(self, path, policy):
        raise NotImplementedError

    def log_train_and_validation_error(self, paths, policy):
        shuffled = np.random.permutation(paths)
        train_size = int(0.8 * len(paths))
        train_paths, validation_paths = shuffled[:train_size], shuffled[train_size:]
        self.fit_model(train_paths)
        train_error = self.calculate_error(train_paths, policy)
        validation_error = self.calculate_error(validation_paths, policy)

        logger.record_tabular(self.title + "TrainError", train_error)
        logger.record_tabular(self.title + "ValidationError", validation_error)
        logger.record_tabular(self.title + "ValTrainDifference", validation_error - train_error)
        logger.record_tabular(self.title + "ValTrainQuotient", validation_error / train_error)

    def calculate_error(self, paths, policy):
        predictions = np.concatenate([self.predict(path, policy) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        return np.linalg.norm(predictions - returns) / len(paths)

    @classmethod
    @autoargs.add_args
    def add_args(cls, parser):
        pass

    @classmethod
    @autoargs.new_from_args
    def new_from_args(cls, args, mdp):
        pass

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        pass
