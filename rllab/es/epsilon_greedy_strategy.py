from rllab.es.base import ExplorationStrategy
from rllab.misc.overrides import overrides
import numpy as np


class EpsilonGreedyStrategy(ExplorationStrategy):
    """
    Exploration strategy that takes a random action with probability epsilon,
    and takes the greedy action with respect to the Q values with probability
    1 - epsilon. This is only suitable for MDPs with discrete actions.
    The value of epsilon can be annealed over time.
    """

    def __init__(
            self,
            mdp,
            max_epsilon=0.1,
            min_epsilon=0.1,
            epsilon_decay_range=1,
            **kwargs):
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_range = epsilon_decay_range

    def get_epsilon(self, t):
        if t > self.epsilon_decay_range:
            return self.min_epsilon
        return self.max_epsilon - (self.max_epsilon - self.min_epsilon) * t / \
            self.epsilon_decay_range

    @overrides
    def get_action(self, t, observation, qfun, **kwargs):
        epsilon = self.get_epsilon(t)
        qval = qfun.compute_qval(observation)
        if np.random.rand() > epsilon:
            return np.argmax(qval)
        else:
            return np.random.randint(len(qval))
