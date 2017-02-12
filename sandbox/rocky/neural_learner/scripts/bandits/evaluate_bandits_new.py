import multiprocessing
import os

from rllab import config
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

import numpy as np
import numba
import csv
import scipy.stats
import sys


def evaluate_thompson_sampling_once(cfg, seed=None):
    n_arms, n_episodes = cfg
    # env = gym.make
    import gym
    env = gym.make(
        'BernoulliBandit-{k}.arms-{n}.episodes-v0'.format(k=n_arms, n=n_episodes),
    )
    env.seed(seed)#np.random.seed(seed)
    # n_arms = env.action_space.n#flat_dim

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


def evaluate_thompson():

    for arms in [5, 10, 50]:

        for n_episodes in [500, 100, 10]:
            # env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)
            cfg = (arms, n_episodes)

            with multiprocessing.Pool() as pool:
                returns = pool.starmap(evaluate_thompson_sampling_once, [(cfg, seed) for seed in range(1000)])
            mean = np.mean(returns)
            std = np.std(returns) / np.sqrt(len(returns) - 1)
            results = (mean, std)

            print(results)

            print(dict(
                n_arms=arms,
                n_episodes=n_episodes,
                avg=results[0],
                stdev=results[1],
            ))

            print("Arms:", arms, flush=True)
            print("N episodes:", n_episodes, flush=True)
            print("Average return:", results[0], flush=True)
            sys.exit()


if __name__ == "__main__":
    evaluate_thompson()
