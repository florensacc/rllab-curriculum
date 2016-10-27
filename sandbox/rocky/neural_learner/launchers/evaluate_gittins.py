from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

import numpy as np
import numba


@numba.njit
def _compute_arm_index(a, b, discount):
    n = a + b
    mu = a / n
    c = np.log(1. / discount)
    return mu + (mu * (1 - mu)) / ((n * ((2 * c + 1. / n) * mu * (1 - mu)) ** 0.5) + mu - 0.5)


class GittinsBernoulli(object):
    def __init__(self):
        self.num_arms = 0
        self.alphas = None
        self.betas = None

    def set_parameters(self, num_arms):
        if num_arms < 2:
            raise ValueError("number of arms must be >= 2")

        self.num_arms = num_arms
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)

    def make_choice(self):
        indices = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            nu = _compute_arm_index(self.alphas[i], self.betas[i], 1.)
            indices[i] = nu

        return np.argmax(indices)

    def save_outcome(self, index, success):
        if success:
            self.alphas[index] += 1
        else:
            self.betas[index] += 1

    def get_means(self):
        return self.alphas / (self.alphas + self.betas)


def evaluate_approx_gittins_once(env):
    env.reset()

    policy = GittinsBernoulli()
    policy.set_parameters(num_arms=env.action_space.flat_dim)

    rewards = []

    for t in range(env.n_episodes):
        action = policy.make_choice()
        _, reward, _, _ = env.step(action)
        policy.save_outcome(index=action, success=reward)
        rewards.append(reward)

    return np.sum(rewards)



if __name__ == "__main__":

    for arms in [5, 10, 50]:

        for n_episodes in [500, 100, 10]:

            env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

            returns = []

            for _ in range(1000):
                ret = evaluate_approx_gittins_once(env)

                returns.append(ret)

                print(np.mean(returns))

            print("Arms:", arms, flush=True)
            print("N episodes:", n_episodes, flush=True)
            print("Average return:", np.mean(returns), flush=True)
