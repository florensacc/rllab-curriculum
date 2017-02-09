import multiprocessing
import random
import os.path
import csv

from rllab import config
from rllab.core.serializable import Serializable
from rllab.misc.console import mkdir_p
from rllab.misc.ext import using_seed
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv, VecMAB
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

import numpy as np
import numba
import scipy.stats

ARMS = [5]
EPISODES = [500]

EPSILONS = [
    (0, 0.01),
    (0.01, 0.05),
    (0.05, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 1.0),
    (0, 1.0),
]


def _compute_approx_arm_index(a, b, discount):
    n = a + b
    mu = a / n
    c = np.log(1. / discount)
    return mu + (mu * (1 - mu)) / ((n * ((2 * c + 1. / n) * mu * (1 - mu)) ** 0.5) + mu - 0.5)


folder_name = "data/iclr2016_prereview"


class EpsilonMABEnv(MABEnv, Serializable):
    def __init__(self, n_arms, epsilon):
        Serializable.quick_init(self, locals())
        self.epsilon = epsilon
        MABEnv.__init__(self, n_arms=n_arms)

    def reset_trial(self):
        while True:
            result = super().reset_trial()
            ranked = np.sort(self.executor.arm_means).flatten()
            best = ranked[-1]
            snd = ranked[-2]
            low, high = self.epsilon
            diff = best - snd
            if low <= diff <= high:
                return result

    def vec_env_executor(self, n_envs):
        return VecEpsilonMAB(n_envs=n_envs, env=self)


class VecEpsilonMAB(VecMAB):
    def step(self, actions, max_path_length):
        next_obs, rewards, dones, env_infos = super().step(actions, max_path_length)
        env_infos["arm_means"] = self.arm_means
        return next_obs, rewards, dones, env_infos

    def reset_trial(self, dones, seeds=None, *args, **kwargs):
        """
        :param dones: array of length == self.n_envs
        :param seeds: array of length == sum(dones)
        :return:
        """
        dones = np.cast['bool'](dones)
        self.ts[dones] = 0
        if self.env.arm_dist == "bernoulli":
            if seeds is None:
                seeds = [None] * len(np.where(dones)[0])
            for idx, seed in zip(np.where(dones)[0], seeds):
                with using_seed(seed):
                    while True:
                        arm_means = np.random.uniform(size=(self.env.n_arms,))
                        ranked = np.sort(arm_means).flatten()
                        best = ranked[-1]
                        snd = ranked[-2]
                        low, high = self.env.epsilon
                        diff = best - snd
                        if low <= diff <= high:
                            break
                    self.arm_means[idx] = arm_means
        else:
            raise NotImplementedError
        return self.reset(dones)


class BanditPolicy(object):
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)

    def save_outcome(self, index, success):
        if success:
            self.alphas[index] += 1
        else:
            self.betas[index] += 1


class GittinsBernoulli(BanditPolicy):
    def make_choice(self):
        indices = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            nu = _compute_approx_arm_index(self.alphas[i], self.betas[i], 1.)
            indices[i] = nu

        return np.argmax(indices)


def memoize(function):
    memo = {}

    def wrapper(*args):
        if args in memo:
            return memo[args]
        else:
            rv = function(*args)
            memo[args] = rv
            return rv

    return wrapper


@memoize
@numba.njit
def _compute_arm_index(a, b, max_k, discount=1.0):
    nu = 0.0
    best_k = 0
    v = np.zeros(max_k)
    for k in range(max_k):
        for j in range(k + 1):
            v[j] = (a + j) / (a + k + b)
        for i in range(k - 1, -1, -1):
            for j in range(i + 1):
                v[j] = (a + j) / (a + i + b) * (1.0 + discount * v[j + 1]) + (b + i - j) / (a + i + b) * discount * \
                                                                             v[j]
        vk = v[0] / (k + 1.0)
        if vk >= nu:
            nu = vk
            best_k = k + 1
    return nu, best_k


class ExactGittinsBernoulli(BanditPolicy):
    def __init__(self, n_arms, max_k=500):
        super().__init__(n_arms)
        self.max_k = max_k

    def make_choice(self):
        indices = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            nu, best_k = _compute_arm_index(self.alphas[i], self.betas[i], self.max_k)
            indices[i] = nu

        return np.argmax(indices)


class Evaluator(object):
    def __init__(self, policy_cls, **policy_args):
        self.policy_cls = policy_cls
        self.policy_args = policy_args

    def __call__(self, env, seed):
        np.random.seed(seed)
        random.seed(seed)
        env.reset()
        policy = self.policy_cls(n_arms=env.action_space.flat_dim, **self.policy_args)
        rewards = []
        actions = []
        for t in range(env.n_episodes):
            action = policy.make_choice()
            _, reward, _, _ = env.step(action)
            policy.save_outcome(index=action, success=reward)
            rewards.append(reward)
            actions.append(action)
        pol_best_arm = np.bincount(actions).argmax()
        true_best_arm = env.wrapped_env.executor.arm_means.flatten().argmax()
        return np.sum(rewards), int(pol_best_arm == true_best_arm)


class ThompsonSampling(BanditPolicy):
    def make_choice(self):
        posterior_means = np.random.beta(self.alphas, self.betas)
        action = np.argmax(posterior_means)
        return action


class OptimisticThompsonSampling(BanditPolicy):
    def __init__(self, n_arms, n_samples=1):
        super().__init__(n_arms)
        self.n_samples = n_samples

    def make_choice(self):
        map_means = None
        map_logprob = None
        for _ in range(self.n_samples):
            posterior_means = np.random.beta(self.alphas, self.betas)
            logprob = np.sum(scipy.stats.beta.logpdf(posterior_means, self.alphas, self.betas))
            if map_logprob is None or logprob > map_logprob:
                map_logprob = logprob
                map_means = posterior_means
        action = np.argmax(map_means)
        return action


class UCBPolicy(BanditPolicy):
    def __init__(self, n_arms, c=0.2):
        super().__init__(n_arms)
        self.c = c
        self.t = 0

    def make_choice(self):
        ms = self.alphas + self.betas
        ks = self.alphas
        if np.any(ms == 0):
            action = np.where(ms == 0)[0][0]
        else:
            indices = ks / ms + self.c * np.sqrt(2 * np.log(self.t) / ms)
            action = np.argmax(indices)
        self.t += 1
        return action


class GreedyPolicy(BanditPolicy):
    def make_choice(self):
        ms = self.alphas + self.betas
        ks = self.alphas
        if np.any(ms == 0):
            action = np.where(ms == 0)[0][0]
        else:
            indices = ks / ms
            action = np.argmax(indices)
        return action


class EpsilonGreedyPolicy(BanditPolicy):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def make_choice(self):
        ms = self.alphas + self.betas
        ks = self.alphas
        if np.any(ms == 0):
            action = np.where(ms == 0)[0][0]
        else:
            if np.random.uniform() < self.epsilon:
                action = np.random.choice(np.arange(self.n_arms))
            else:
                indices = ks / ms
                action = np.argmax(indices)
        return action


class RandomPolicy(BanditPolicy):
    def make_choice(self):
        return np.random.choice(np.arange(self.n_arms))


def evaluate(strategy, name):
    mkdir_p(os.path.join(config.PROJECT_PATH, folder_name))
    csv_file = os.path.join(config.PROJECT_PATH, folder_name, name + ".csv")

    with open(csv_file, "w") as f:
        writer = csv.DictWriter(
            f,
            ["strategy", "n_arms", "n_episodes", "avg", "stdev", "epsilon_from", "epsilon_to", "best_arm_percent"]
        )
        writer.writeheader()

        for arms in ARMS:

            for n_episodes in EPISODES:

                for epsilon in EPSILONS:
                    env = MultiEnv(
                        wrapped_env=EpsilonMABEnv(n_arms=arms, epsilon=epsilon),
                        n_episodes=n_episodes,
                        episode_horizon=1,
                        discount=1
                    )

                    with multiprocessing.Pool() as pool:
                        returns_best_arms = pool.starmap(strategy, [(env, seed) for seed in range(1000)])
                    returns = [x[0] for x in returns_best_arms]
                    mean = np.mean(returns)
                    std = np.std(returns) / np.sqrt(len(returns) - 1)
                    best_arm_ptg = np.mean([x[1] for x in returns_best_arms])

                    print("Strategy: ", name)
                    print("Arms:", arms, flush=True)
                    print("N episodes:", n_episodes, flush=True)
                    print("Average return:", mean, flush=True)
                    print("Std return:", std, flush=True)
                    print("Epsilon range:", epsilon, flush=True)
                    print("%Best arm:", best_arm_ptg, flush=True)

                    writer.writerow(dict(
                        strategy=name,
                        n_arms=arms,
                        n_episodes=n_episodes,
                        avg=mean,
                        stdev=std,
                        epsilon_from=epsilon[0],
                        epsilon_to=epsilon[1],
                        best_arm_percent=best_arm_ptg,
                    ))


if __name__ == "__main__":
    evaluate(Evaluator(UCBPolicy, c=0.2), "ucb")
    evaluate(Evaluator(UCBPolicy, c=1.0), "ucb_default")
    evaluate(Evaluator(ThompsonSampling), "ts")
    evaluate(Evaluator(OptimisticThompsonSampling, n_samples=7), "ots")
    evaluate(Evaluator(GittinsBernoulli), "gittins")
    evaluate(Evaluator(GreedyPolicy), "greedy")
    evaluate(Evaluator(EpsilonGreedyPolicy, epsilon=0.), "egreedy")
    evaluate(Evaluator(RandomPolicy), "random")
