import numpy as np


class GprPolicyWrapper(object):
    def __init__(self, gpr_policy, p_rand_action=0., stds=None):
        self.gpr_policy = gpr_policy
        self.p_rand_action = p_rand_action
        self.stds = stds

    def get_action(self, obs):
        action = self.gpr_policy.get_action(obs)
        is_rand_action = False
        if self.stds is not None:
            if np.random.uniform() < self.p_rand_action:
                is_rand_action = True
                action = action + np.random.normal(size=len(action)) * self.stds
        return action, dict(is_rand_action=is_rand_action)

    def reset(self):
        self.gpr_policy.reset()

    def set_param_values(self, params):
        pass
