from rllab.policies.base import Policy
from sandbox.rocky.hogwild.shared_parameterized import SharedParameterized


class SharedPolicy(Policy, SharedParameterized):
    def __init__(self, policy):
        SharedParameterized.__init__(self, policy)
