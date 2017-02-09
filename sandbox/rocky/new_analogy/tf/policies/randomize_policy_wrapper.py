import numpy as np


class RandomizePolicyWrapper(object):

    def __init__(self, policy, action_mean, action_std, noise_scale, p_random_action):
        self.policy = policy
        self.action_mean = action_mean
        self.action_std = action_std
        self.noise_scale = noise_scale
        self.p_random_action = p_random_action

    def get_action(self, obs):
        action = self.policy.get_action(obs)[0]
        if np.random.uniform() < self.p_random_action:
            action = self.action_mean + np.random.normal(size=len(action)) * self.action_std
        else:
            action += np.random.normal(size=len(action)) * self.action_std * self.noise_scale
        return action, dict()

    def reset(self):
        self.policy.reset()

    def set_param_values(self, params):
        pass
