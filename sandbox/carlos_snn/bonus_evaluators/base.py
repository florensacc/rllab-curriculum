class BonusEvaluator(object):

    def fit_before_process_samples(self, paths):
        """
        NEEDED: Called in process_samples, before processing them. This initializes the hashes based on the current obs.
        """
        raise NotImplementedError

    def predict(self, path):
        """
        NEEDED: Gives the bonus!
        :param path: reward computed path by path
        :return: a 1d array
        """
        raise NotImplementedError

    def fit_after_process_samples(self, samples_data):
        """
        NEEDED
        """
        pass

    def log_diagnostics(self, paths):
        """
        NEEDED
        """
        pass
