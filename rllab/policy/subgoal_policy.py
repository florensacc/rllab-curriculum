from rllab.policy.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
import numpy as np


class SubgoalPolicy(StochasticPolicy, LasagnePowered, Serializable):

    """
    The high-level policy receives the raw observation, and emits a subgoal
    for the low-level policy. The low-level policy receives the raw observation
    concatenated with the subgoal, and emits the actual control for the MDP.
    """

    def __init__(self, mdp, high_policy, low_policy):
        Serializable.quick_init(self, locals())
        super(SubgoalPolicy, self).__init__(mdp)
        self._high_policy = high_policy
        self._low_policy = low_policy
        LasagnePowered.__init__(self, self._high_policy.output_layers + self._low_policy.output_layers)

    @property
    def high_policy(self):
        return self._high_policy

    @property
    def low_policy(self):
        return self._low_policy

    def get_action(self, observation):
        # First, sample a goal
        subgoal, high_pdist = self._high_policy.get_action(observation)
        action, low_pdist = self._low_policy.get_action(np.concatenate([
            observation.flatten(),
            subgoal.flatten(),
        ]))
        return action, np.concatenate([high_pdist, low_pdist])

    def compute_entropy(self, pdist):
        return np.nan
