from sandbox.carlos_precond.algos.npo import NPO  # only change in NPO
from sandbox.carlos_precond.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer_precond  # different optimizers
from rllab.core.serializable import Serializable


class TRPO(NPO, Serializable):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer_precond(**optimizer_args)  # main diff
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)
