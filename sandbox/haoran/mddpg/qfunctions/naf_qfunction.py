import abc
from predictors.state_action_network import StateActionNetwork
from qfunctions.optimizable_q_function import OptimizableQFunction
from qfunctions.separable_q_function import SeparableQFunction


class NAFQFunction(StateActionNetwork,
                   OptimizableQFunction,
                   SeparableQFunction):
    @abc.abstractmethod
    def _create_network(self):
        return

    @abc.abstractmethod
    def get_implicit_advantage_function(self):
        return

    @abc.abstractmethod
    def get_implicit_policy(self):
        return

    @abc.abstractmethod
    def get_implicit_value_function(self):
        return
