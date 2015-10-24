import numpy as np
from base import ValueFunction
from misc.overrides import overrides

class NoValueFunction(ValueFunction):

    @overrides
    def get_param_values(self):
        return None

    @overrides
    def set_param_values(self, val):
        pass

    @overrides
    def fit(self, paths):
        pass

    @overrides
    def predict(self, path):
        return np.zeros_like(path["returns"])
