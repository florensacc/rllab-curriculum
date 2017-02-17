import numpy as np

class Annealer(object):
    def __init__(self, init_value, final_value, n_iter):
        self.init_value = init_value
        self.final_value = final_value
        self.n_iter = n_iter

    def get_new_value(self, iteration):
        raise NotImplementedError

class LinearAnnealer(Annealer):
    """
    Linearly anneal until a specific iteration
    """
    def __init__(self, init_value, final_value, n_iter, stop_iter):
        super().__init__(init_value, final_value, n_iter)
        self.stop_iter = stop_iter
        assert stop_iter >= 0 and stop_iter < n_iter

    def get_new_value(self, iteration):
        assert iteration < self.n_iter # iteration starts from 0
        if iteration < self.stop_iter:
            b = self.init_value
            a = (self.final_value - self.init_value) / self.stop_iter
            value = a * iteration + b
        else:
            value = self.final_value
        return value

class LogLinearAnnealer(Annealer):
    """
    Linearly anneal until a specific iteration
    """
    def __init__(self, init_value, final_value, n_iter, stop_iter):
        super().__init__(init_value, final_value, n_iter)
        self.stop_iter = stop_iter
        assert stop_iter >= 0 and stop_iter < n_iter

    def get_new_value(self, iteration):
        assert iteration < self.n_iter # iteration starts from 0
        if iteration < self.stop_iter:
            b = np.log(self.init_value)
            a = (np.log(self.final_value) - np.log(self.init_value)) / self.stop_iter
            value = np.exp(a * iteration + b)
        else:
            value = self.final_value
        return value
