from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.base import StochasticPolicy
from rllab.core.serializable import Serializable



class StochasticGRUPolicy(StochasticPolicy, Serializable):
    """
    Structure the hierarchical policy as a recurrent network with stochastic gated recurrent unit, where the
    stochastic component of the hidden state will play the role of internal goals. Binary (or continuous) decision
    gates control the updates to the internal goals.
    """

    def __init__(self, env_spec, subgoal_spec):
        Serializable.quick_init(self, locals())

    def reset(self):
        self.hidden_state =

    def get_action(self, observation):
        pass



