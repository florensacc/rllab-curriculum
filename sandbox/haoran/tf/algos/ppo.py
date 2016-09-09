from __future__ import print_function
from __future__ import absolute_import

from sandbox.haoran.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from sandbox.haoran.tf.algos.npo import NPO


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
            optimizer = PenaltyLbfgsOptimizer(**dict(dict(name="optimizer"), **optimizer_args))
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)
