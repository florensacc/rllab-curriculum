from rllab.algo.npo import NPO
from rllab.optimizer.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class TRPO(NPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            **kwargs):
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer(**kwargs)
        super(TRPO, self).__init__(optimizer=optimizer, **kwargs)
