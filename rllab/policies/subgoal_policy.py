from rllab.policies.base import StochasticPolicy
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
        self._subgoal_space = env_spec.subgoal_space
        self._high_policy = high_policy
        self._low_policy = low_policy
        self._subgoal_interval = subgoal_interval
        self._interval_counter = 0
        self._subgoal = None
        self._high_agent_info = None
        self.reset()
        LasagnePowered.__init__(self, self._high_policy.output_layers + self._low_policy.output_layers)

    @property
    def subgoal_space(self):
        return self._subgoal_space

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
        self._high_agent_info = None

    @property
    def subgoal_interval(self):
        return self._subgoal_interval

    def get_action(self, observation):
        self._interval_counter += 1
        high_obs = observation
        if self._interval_counter >= self._subgoal_interval:
            # update subgoal
            self._subgoal, self._high_agent_info = self._high_policy.get_action(high_obs)
            # reset counter
            self._interval_counter = 0
        low_obs = (high_obs, self._subgoal)
        action, low_agent_info = self._low_policy.get_action(low_obs)
        return action, dict(
            high=self._high_agent_info,
            low=low_agent_info,
            subgoal=self.subgoal_space.flatten(self._subgoal),
            high_obs=self.high_policy.observation_space.flatten(high_obs),
            low_obs=self.low_policy.observation_space.flatten(low_obs),
        )

    # def compute_entropy(self, pdist):
    #     return np.nan

    # def split_pdists(self, pdists):
    #     high_pdist_dim = self.high_policy.pdist_dim
    #     low_pdist_dim = self.low_policy.pdist_dim
    #     return np.split(pdists, [high_pdist_dim, high_pdist_dim + low_pdist_dim], axis=1)
