from rllab.optimizer.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.algo.npo import NPO


class PPO(NPO):
    """
    Penalized Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)
