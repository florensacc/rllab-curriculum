from .base import Policy
from misc.overrides import overrides
import numpy as np

class EpsilonGreedyPolicy(Policy):

    def __init__(self, qfunc, epsilon):
        self._qfunc = qfunc
        self.epsilon = epsilon

    @overrides
    def get_actions(self, observations):
        qval = self._qfunc.compute_qval(observations)
        n_actions = qval.shape[1]
        N = len(observations)
        rnd = np.random.rand(N)
        use_greedy = rnd > self.epsilon
        actions = use_greedy * np.argmax(qval, axis=1) + \
                (1 - use_greedy) * np.random.choice(n_actions, size=N)
        pdists = np.zeros((N, 0))
        return actions, pdists

    @overrides
    def get_param_values(self):
        return np.append(self._qfunc.get_param_values(), self.epsilon)

    @overrides
    def set_param_values(self, flattened_params):
        self.epsilon = flattened_params[-1]
        self._qfunc.set_param_values(flattened_params[:-1])
