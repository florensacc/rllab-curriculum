# from rllab.algos.npo import NPO
from sandbox.carlos_snn.npo_snn import NPO_snn
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable


class TRPO_snn(NPO_snn, Serializable):
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
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TRPO_snn, self).__init__(optimizer=optimizer, **kwargs)
