from .base import DiscretePolicy

class DiscreteProxyPolicy(DiscretePolicy):

    def __init__(self, base_policy):
        self._base_policy = base_policy

    def compute_action_probs(self, states):
        return self._base_policy.compute_action_probs(states)

    def get_actions(self, states):
        return self._base_policy.get_actions(states)

    def get_param_values(self):
        return self._base_policy.get_param_values()

    def set_param_values(self, flattened_parameters):
        self._base_policy.set_param_values(flattened_parameters)

    @property
    def params(self):
        return self._base_policy.params
