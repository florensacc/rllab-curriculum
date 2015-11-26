from rllab.qfun.base import ContinuousQFunction
from rllab.qfun.lasagne_q_function import LasagneQFunction


class ContinuousNNQFunction(ContinuousQFunction, LasagnePowered):

    def get_qvals_sym(self, input_var, actions_var):
        pass
