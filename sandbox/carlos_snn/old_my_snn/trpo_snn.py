# from rllab.algos.npo import NPO
from rllab.core.serializable import Serializable
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer

from sandbox.carlos_snn.old_my_snn.npo_snn_rewardMI import NPO_snn as NPO_snn
# from sandbox.carlos_snn.old_my_snn.npo_snn_rewardKL import NPO_snn_kl as NPO_snn
# from sandbox.carlos_snn.old_my_snn.npo_snn_rewardL2 import NPO_snn_l2 as NPO_snn


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
