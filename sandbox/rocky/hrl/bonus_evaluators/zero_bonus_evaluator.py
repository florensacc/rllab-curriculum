from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.hrl.bonus_evaluators.base import BonusEvaluator
from rllab.core.serializable import Serializable
import numpy as np
import theano.tensor as TT


class ZeroBonusEvaluator(BonusEvaluator, Serializable):
    def __init__(self, env_spec, policy):
        Serializable.quick_init(self, locals())
        self.env_spec = env_spec
        self.policy = policy

    def fit(self, paths):
        pass

    def predict(self, path):
        return np.zeros_like(path["rewards"])

    def log_diagnostics(self, paths):
        pass

    def bonus_sym(self, raw_obs_var, action_var, state_info_vars):
        return TT.zeros(raw_obs_var.shape[:-1])
