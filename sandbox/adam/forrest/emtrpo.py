from sandbox.adam.forrest.emnpo import EMNPO
from sandbox.adam.forrest.conjugate_gradient_optimizer2 import ConjugateGradientOptimizer2
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable


class EMTRPO(EMNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer2(**optimizer_args)
        super(EMTRPO, self).__init__(optimizer=optimizer, **kwargs)
