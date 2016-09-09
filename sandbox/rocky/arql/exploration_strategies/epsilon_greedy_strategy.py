

import numpy as np


class EpsilonGreedyStrategy(object):
    def __init__(
            self,
            env_spec,
            max_epsilon=1.,
            min_epsilon=0.1,
            epsilon_decay_range=1000000,
    ):
        self.env_spec = env_spec
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_range = epsilon_decay_range

    def reset(self):
        pass

    def get_action(self, iteration, observation, policy):

        epsilon = np.clip(
            np.interp(iteration, [0, self.epsilon_decay_range], [self.max_epsilon, self.min_epsilon]),
            self.min_epsilon,
            self.max_epsilon
        )
        if np.random.uniform() < epsilon:
            action = self.env_spec.action_space.sample()
        else:
            action, _ = policy.get_action(observation)
        return action
