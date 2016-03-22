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

    def __init__(self, env_spec, high_policy, low_policy, subgoal_interval=1):
        """
        :param env_spec: Spec for the MDP
        :param high_policy: High-level policy, which feeds subgoal to the low-level policy
        :param low_policy: Low-level policy
        :param subgoal_interval: The time interval between each high-level decision. It defaults to 1.
        :return:
        """
        Serializable.quick_init(self, locals())
        super(SubgoalPolicy, self).__init__(env_spec)
        self._high_policy = high_policy
        self._low_policy = low_policy
        self._subgoal_interval = subgoal_interval
        self._interval_counter = 0
        self._subgoal = None
        self._high_pdist = None
        self.reset()
        LasagnePowered.__init__(self, self._high_policy.output_layers + self._low_policy.output_layers)

    @property
    def high_policy(self):
        return self._high_policy

    @property
    def low_policy(self):
        return self._low_policy

    def reset(self):
        # Set the counter so that the subgoal will be sampled on the first time step
        self._interval_counter = self._subgoal_interval - 1
        self._subgoal = None
        self._high_pdist = None

    @property
    def subgoal_interval(self):
        return self._subgoal_interval

    def act(self, observation):
        self._interval_counter += 1
        if self._interval_counter >= self._subgoal_interval:
            # update subgoal
            self._subgoal, self._high_pdist = self._high_policy.act(observation)
            # reset counter
            self._interval_counter = 0
        action, low_pdist = self._low_policy.act(np.concatenate([
            observation.flatten(),
            self._subgoal.flatten(),
        ]))
        return action, np.concatenate([self._high_pdist, low_pdist, self._subgoal])

    def compute_entropy(self, pdist):
        return np.nan

    def split_pdists(self, pdists):
        high_pdist_dim = self.high_policy.pdist_dim
        low_pdist_dim = self.low_policy.pdist_dim
        return np.split(pdists, [high_pdist_dim, high_pdist_dim + low_pdist_dim], axis=1)
