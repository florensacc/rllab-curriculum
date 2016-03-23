from rllab.algo.npo import NPO
from rllab.optimizer.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.misc import ext


class TNPG(NPO):
    """
    Truncated Natural Policy Gradient.
    """

    def __init__(
            self,
            optimizer=None,
            **kwargs):
        if optimizer is None:
            optimizer_args = ext.merge_dict(
                dict(
                    max_backtracks=0,
                ),
                kwargs
            )
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TNPG, self).__init__(optimizer=optimizer, **kwargs)
