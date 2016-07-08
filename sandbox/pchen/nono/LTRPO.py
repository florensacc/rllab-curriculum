from rllab.algos.npo import NPO
from rllab.misc.overrides import overrides
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable

policy = None

class LTRPO(NPO):
    """
    leaky Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(LTRPO, self).__init__(optimizer=optimizer, **kwargs)

    @overrides
    def optimize_policy(self, itr, samples_data):
        global policy
        policy = self.policy
        return super(LTRPO, self).optimize_policy(itr, samples_data)
