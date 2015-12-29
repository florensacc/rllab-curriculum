from rllab.es.base import ExplorationStrategy
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
import numpy as np


class GaussianStrategy(ExplorationStrategy):
    """
    Exploration strategy that takes as input a deterministic policy, and
    adds uncorrelated Gaussian noise to the action that the policy chooses.
    The standard deviation (sigma) of the noise can be annealed over time.
    """

    @autoargs.arg('max_sigma', type=float,
                  help='Initial value of sigma')
    @autoargs.arg('min_sigma', type=float,
                  help='Minimal value of sigma')
    @autoargs.arg('sigma_decay_range', type=int,
                  help='How many samples should it take for sigma to decay '
                       'from max_sigma to min_sigma')
    def __init__(
            self,
            mdp,
            max_sigma=0.01,
            min_sigma=0.01,
            sigma_decay_range=1,
            **kwargs):
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.sigma_decay_range = sigma_decay_range

    def get_sigma(self, t):
        if t > self.sigma_decay_range:
            return self.min_sigma
        return self.max_sigma - (self.max_sigma - self.min_sigma) * t / \
            self.sigma_decay_range

    @overrides
    def get_action(self, t, observation, policy, **kwargs):
        sigma = self.get_sigma(t)
        action, _ = policy.get_action(observation)
        return action + np.random.randn(len(action)) * sigma
