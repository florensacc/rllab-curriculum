from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.algo.npo import NPO


class PPO(NPO):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            **kwargs):
        if optimizer is None:
            optimizer = PenaltyLbfgsOptimizer(**kwargs)
            super(PPO, self).__init__(optimizer=optimizer, **kwargs)
