from sandbox.rocky.tf.policies.base import Policy
import numpy as np


def _compute_arm_index(a, b, discount=1.0):
    n = a + b
    mu = a / n
    c = np.log(1. / discount)
    return mu + (mu * (1 - mu)) / ((n * ((2 * c + 1. / n) * mu * (1 - mu)) ** 0.5) + mu - 0.5)


class ApproximateGittinsPolicy(Policy):
    def __init__(self, env_spec):
        n_arms = env_spec.action_space.n
        self.n_arms = n_arms
        self.n_envs = 1
        self.alphas = np.ones((1, self.n_arms))
        self.betas = np.ones((1, self.n_arms))
        self.ts = np.zeros((1,))

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.cast['bool'](dones)
        if self.n_envs != len(dones):
            self.n_envs = len(dones)
            self.alphas = np.ones((self.n_envs, self.n_arms))
            self.betas = np.ones((self.n_envs, self.n_arms))
            self.ts = np.zeros((self.n_envs,))
        else:
            self.alphas[dones] = 1
            self.betas[dones] = 1
            self.ts[dones] = 0

    def get_actions(self, observations):
        observations = np.asarray(observations)
        last_rewards = observations[:, 2]
        inc_alpha = np.logical_and(self.ts, last_rewards)
        inc_beta = np.logical_and(self.ts, np.logical_not(last_rewards))
        last_actions = observations[:, 1]
        if np.any(inc_alpha):
            self.alphas[inc_alpha, last_actions[inc_alpha]] += 1
        if np.any(inc_beta):
            self.betas[inc_beta, last_actions[inc_beta]] += 1
        indices = _compute_arm_index(self.alphas, self.betas, discount=1.)
        self.ts += 1
        return np.argmax(indices, axis=1), dict()
