from __future__ import print_function
from __future__ import absolute_import


class BonusEvaluator(object):

    def fit(self, paths):
        raise NotImplementedError

    def predict(self, path):
        raise NotImplementedError

    def log_diagnostics(self, paths):
        pass

    def bonus_sym(self, raw_obs_var, action_var, state_info_vars):
        raise NotImplementedError
