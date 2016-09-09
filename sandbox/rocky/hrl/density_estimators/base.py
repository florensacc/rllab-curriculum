



class DensityEstimator(object):

    def fit(self, xs):
        """
        Fit the density model given the data.
        :param xs: should have dimension N*D, where N is the number of samples
        """
        raise NotImplementedError

    def predict_log_likelihood(self, xs):
        """
        Return the log likelihood of each data entry.
        """
        raise NotImplementedError

    def log_likelihood_sym(self, x_var):
        """
        Return the symbolic log likelihood of each data entry.
        """
        raise NotImplementedError

