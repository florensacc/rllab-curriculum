from rllab.q_functions.base import QFunction
from sandbox.rocky.hogwild.shared_parameterized import SharedParameterized


class SharedQFunction(QFunction, SharedParameterized):

    def __init__(self, qf):
        SharedParameterized.__init__(self, qf)
