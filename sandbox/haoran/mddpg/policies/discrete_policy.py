import tensorflow as tf
import numpy as np

from rllab.core.serializable import Serializable
from rllab.policies.base import Policy


class DiscretePolicy(Policy):
    """
    A policy that discretizes the state and action space, choosing an action
    by drawing from the discrete action choices at the discrete state closest
    to the current
    """
    def __init__(
        self,
        scope_name,
        discrete_policy,
        observation_list,
        action_list,
        **kwargs
    ):
        """
        :param observation_list: a (None, obs_dim) array
        """
        Serializable.quick_init(self, locals())
        self.discrete_policy = discrete_policy
        self.observation_list = observation_list
        self.ns, self.ds = self.observation_list.shape
        self.action_list = action_list
        self.na, self.da = self.action_list.shape

    def get_action(self, observation):
        return self.sess.run(self.output,
                             {self.observations_placeholder: [observation]}), {}

    def get_actions(self, observations):
        N = observations.shape[0]
        obs_expanded = np.repeat(
            observations,
            self.ns,
            axis=0
        )
        obs_list_expanded = np.outer(
            np.ones(N),
            self.observation_list).reshape((-1,self.ds))
        diff = np.sum((obs_expanded - obs_list_expanded)**2, axis=1)
        obs_indices = np.argmin(diff.reshape((N, self.ns)), axis=1)
        return obs_indices
