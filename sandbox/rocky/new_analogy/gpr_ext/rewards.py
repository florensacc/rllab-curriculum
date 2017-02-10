from gpr.reward import Reward, residual2reward
import numpy as np


class SenseDistReward(Reward):

    def __init__(self, key, target, metric="L2"):
        self.key = key
        self.target = target
        self.metric = metric

    def compute_reward(self, s):
        return residual2reward(
            np.reshape(s[self.key].flatten() - self.target, (1, -1)),
            self.metric
        )


class TimeReward(Reward):

    # def __init__(self):
    #     self.key = key
    #     self.target = target
    #     self.metric = metric

    def compute_reward(self, s):
        return s["t"]#residual2reward(
        #     np.reshape(s[self.key].flatten() - self.target, (1, -1)),
        #     self.metric
        # )
