

from rllab.algos.base import RLAlgorithm


class SupervisedBonusTrainer(RLAlgorithm):
    """
    This algorithm fits a policy to a set of supervised training samples.
    In essence, it maximizes E[log(p_pi(tau))]
    """

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy

    def init_opt(self):
        pass

    def train(self):
        self.init_opt()
