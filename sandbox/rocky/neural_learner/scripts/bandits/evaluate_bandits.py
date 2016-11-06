import multiprocessing
import os

from rllab import config
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

import numpy as np
import numba
import csv
import scipy.stats


def _compute_approx_arm_index(a, b, discount):
    n = a + b
    mu = a / n
    c = np.log(1. / discount)
    return mu + (mu * (1 - mu)) / ((n * ((2 * c + 1. / n) * mu * (1 - mu)) ** 0.5) + mu - 0.5)


folder_name = "iclr2016_new"


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
            nu = _compute_approx_arm_index(self.alphas[i], self.betas[i], 1.)
            indices[i] = nu

        return np.argmax(indices)

    def save_outcome(self, index, success):
        if success:
            self.alphas[index] += 1
        else:
            self.betas[index] += 1

    def get_means(self):
        return self.alphas / (self.alphas + self.betas)


def evaluate_approx_gittins_once(env, seed=None):
    np.random.seed(seed)
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


class ExactGittinsBernoulli(object):
    def __init__(self, max_k=500):
        self.num_arms = 0
        self.alphas = None
        self.betas = None
        self.max_k = max_k

    def set_parameters(self, num_arms):
        if num_arms < 2:
            raise ValueError("number of arms must be >= 2")

        self.num_arms = num_arms
        self.alphas = np.ones(self.num_arms)
        self.betas = np.ones(self.num_arms)

    def make_choice(self):  # , horizon):
        # if np.sum(self.betas)+np.sum(self.alphas) < 3*self.num_arms:
        #     for i in range(self.num_arms):
        #         if self.betas[i]+self.alphas[i] < 3:
        #             return i

        indices = np.zeros(self.num_arms)
        for i in range(self.num_arms):
            nu, best_k = _compute_arm_index(self.alphas[i], self.betas[i], self.max_k)  # min(self.max_k, horizon))
            indices[i] = nu

        return np.argmax(indices)

    # def make_choice(self):
    #     indices = np.zeros(self.num_arms)
    #     for i in range(self.num_arms):
    #         nu = _compute_arm_index(self.alphas[i], self.betas[i], 1.)
    #         indices[i] = nu
    #
    #     return np.argmax(indices)

    def save_outcome(self, index, success):
        if success:
            self.alphas[index] += 1
        else:
            self.betas[index] += 1

    def get_means(self):
        return self.alphas / (self.alphas + self.betas)


def evaluate_gittins_once(env, seed=None):
    np.random.seed(seed)
    env.reset()

    policy = ExactGittinsBernoulli()
    policy.set_parameters(num_arms=env.action_space.flat_dim)

    rewards = []

    for t in range(env.n_episodes):
        action = policy.make_choice()
        _, reward, _, _ = env.step(action)
        policy.save_outcome(index=action, success=reward)
        rewards.append(reward)

    return np.sum(rewards)


def evaluate_thompson_sampling_once(env, seed=None):
    np.random.seed(seed)
    n_arms = env.action_space.flat_dim

    alphas = np.ones((n_arms,))
    betas = np.ones((n_arms,))

    rewards = []

    env.reset()

    for t in range(env.n_episodes):
        posterior_means = np.random.beta(alphas, betas)
        action = np.argmax(posterior_means)
        _, reward, _, _ = env.step(action)
        if reward > 0:
            alphas[action] += 1
        else:
            betas[action] += 1
        rewards.append(reward)
    return np.sum(rewards)


def evaluate_optimistic_thompson_once(env, n_samples=1, seed=None):
    np.random.seed(seed)
    n_arms = env.action_space.flat_dim

    alphas = np.ones((n_arms,))
    betas = np.ones((n_arms,))

    rewards = []

    env.reset()

    for t in range(env.n_episodes):
        map_means = None
        map_logprob = None
        for _ in range(n_samples):
            posterior_means = np.random.beta(alphas, betas)
            logprob = np.sum(scipy.stats.beta.logpdf(posterior_means, alphas, betas))
            if map_logprob is None or logprob > map_logprob:
                map_logprob = logprob
                map_means = posterior_means
        action = np.argmax(map_means)
        _, reward, _, _ = env.step(action)
        if reward > 0:
            alphas[action] += 1
        else:
            betas[action] += 1
        rewards.append(reward)
    return np.sum(rewards)


def evaluate_ucb_once(env, c=1., seed=None):
    np.random.seed(seed)
    n_arms = env.action_space.flat_dim

    ks = np.ones((n_arms,))
    ms = np.ones((n_arms,)) * 2

    rewards = []

    env.reset()

    for t in range(env.n_episodes):
        if np.any(ms == 0):
            action = np.where(ms == 0)[0][0]
        else:
            indices = ks / ms + c * np.sqrt(2 * np.log(t) / ms)
            action = np.argmax(indices)
        _, reward, _, _ = env.step(action)
        if reward > 0:
            ks[action] += 1
        ms[action] += 1
        rewards.append(reward)
    return np.sum(rewards)


def evaluate_greedy_once(env, seed=None):
    np.random.seed(seed)
    n_arms = env.action_space.flat_dim

    ks = np.ones((n_arms,))
    ms = np.ones((n_arms,)) * 2

    rewards = []

    env.reset()

    for t in range(env.n_episodes):
        if np.any(ms == 0):
            action = np.where(ms == 0)[0][0]
        else:
            indices = ks / ms
            action = np.argmax(indices)
        _, reward, _, _ = env.step(action)
        if reward > 0:
            ks[action] += 1
        ms[action] += 1
        rewards.append(reward)
    return np.sum(rewards)


def evaluate_epsilon_greedy_once(env, epsilon=0.1, seed=None):
    np.random.seed(seed)
    n_arms = env.action_space.flat_dim

    ks = np.ones((n_arms,))
    ms = np.ones((n_arms,)) * 2

    rewards = []

    env.reset()

    for t in range(env.n_episodes):
        if np.any(ms == 0):
            action = np.where(ms == 0)[0][0]
        else:
            if np.random.uniform() < epsilon:
                action = np.random.choice(np.arange(n_arms))
            else:
                indices = ks / ms
                action = np.argmax(indices)
        _, reward, _, _ = env.step(action)
        if reward > 0:
            ks[action] += 1
        ms[action] += 1
        rewards.append(reward)
    return np.sum(rewards)


def evaluate_random_once(env, seed=None):
    np.random.seed(seed)
    n_arms = env.action_space.flat_dim
    rewards = []
    env.reset()

    for t in range(env.n_episodes):
        action = np.random.choice(np.arange(n_arms))
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
    return np.sum(rewards)


def evaluate_ucb():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/ucb1_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev", "best_c"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:

                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                best_mean = None
                best_results = None

                for c in range(11):
                    c = c * 0.1
                    with multiprocessing.Pool() as pool:
                        returns = pool.starmap(evaluate_ucb_once, [(env, c, seed) for seed in range(1000)])
                    mean = np.mean(returns)
                    std = np.std(returns) / np.sqrt(len(returns) - 1)
                    results = (c, mean, std)

                    if best_mean is None or mean > best_mean:
                        best_mean = mean
                        best_results = results

                    print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=best_results[1],
                    stdev=best_results[2],
                    best_c=best_results[0]
                ))
                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", best_results[1], flush=True)


def evaluate_thompson():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/thompson_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:
                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_thompson_sampling_once, [(env, seed) for seed in range(1000)])
                mean = np.mean(returns)
                std = np.std(returns) / np.sqrt(len(returns) - 1)
                results = (mean, std)

                print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=results[0],
                    stdev=results[1],
                ))
                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", results[0], flush=True)


def evaluate_gittins():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/gittins_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [10, 100, 500]:
                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                # returns = []

                # for seed in range(1000):
                #     ret = evaluate_gittins_once(env, seed)
                #     returns.append(ret)
                #     print(np.mean(returns))

                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_gittins_once, [(env, seed) for seed in range(1000)])
                mean = np.mean(returns)
                std = np.std(returns) / np.sqrt(len(returns) - 1)
                results = (mean, std)

                print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=results[0],
                    stdev=results[1],
                ))

                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", results[0], flush=True)


def evaluate_approx_gittins():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/approx_gittins_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:
                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_approx_gittins_once, [(env, seed) for seed in range(1000)])
                mean = np.mean(returns)
                std = np.std(returns) / np.sqrt(len(returns) - 1)
                results = (mean, std)

                print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=results[0],
                    stdev=results[1],
                ))

                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", results[0], flush=True)


def evaluate_greedy():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/greedy_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:
                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_greedy_once, [(env, seed) for seed in range(1000)])
                mean = np.mean(returns)
                std = np.std(returns) / np.sqrt(len(returns) - 1)
                results = (mean, std)

                print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=results[0],
                    stdev=results[1],
                ))

                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", results[0], flush=True)


def evaluate_epsilon_greedy():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/epsilon_greedy_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev", "best_epsilon"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:

                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                best_mean = None
                best_results = None

                for epsilon in range(11):
                    epsilon = epsilon * 0.1
                    with multiprocessing.Pool() as pool:
                        returns = pool.starmap(evaluate_epsilon_greedy_once, [(env, epsilon, seed) for seed in range(
                            1000)])
                    mean = np.mean(returns)
                    std = np.std(returns) / np.sqrt(len(returns) - 1)
                    results = (epsilon, mean, std)

                    if best_mean is None or mean > best_mean:
                        best_mean = mean
                        best_results = results

                    print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=best_results[1],
                    stdev=best_results[2],
                    best_epsilon=best_results[0]
                ))
                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", best_results[1], flush=True)


def evaluate_optimistic_thompson():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/optimistic_thompson_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev", "best_n_samples"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:

                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                best_mean = None
                best_results = None

                for n_samples in range(1, 21):
                    with multiprocessing.Pool() as pool:
                        returns = pool.starmap(evaluate_optimistic_thompson_once,
                                               [(env, n_samples, seed) for seed in range(1000)])
                    mean = np.mean(returns)
                    std = np.std(returns) / np.sqrt(len(returns) - 1)
                    results = (n_samples, mean, std)

                    if best_mean is None or mean > best_mean:
                        best_mean = mean
                        best_results = results

                    print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=best_results[1],
                    stdev=best_results[2],
                    best_n_samples=best_results[0]
                ))
                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", best_results[1], flush=True)


def evaluate_random():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/random_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [500, 100, 10]:
                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)

                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_random_once, [(env, seed) for seed in range(1000)])
                mean = np.mean(returns)
                std = np.std(returns) / np.sqrt(len(returns) - 1)
                results = (mean, std)

                print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=results[0],
                    stdev=results[1],
                ))

                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", results[0], flush=True)


if __name__ == "__main__":
    evaluate_ucb()
    # evaluate_thompson()
    # evaluate_approx_gittins()
    # evaluate_greedy()
    # evaluate_epsilon_greedy()
    # evaluate_random()
    # evaluate_optimistic_thompson()

    # evaluate_gittins()
