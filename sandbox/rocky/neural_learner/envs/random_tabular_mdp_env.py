# Code adopted from https://github.com/iosband/TabulaRL/blob/master/src/finite_tabular_agents.py
# It would actually be extremely cool if trained over random MDPs, and then tested on these benchmarks,
# it could perform well...

import numpy as np

from rllab.envs.base import Step, Env
from rllab.misc import special
from rllab.misc.ext import using_seed
from rllab.spaces import Discrete


class RandomTabularMDPEnv(Env):
    def __init__(
            self, n_states, n_actions, alpha0=1., mu0=1., tau0=1., tau=1.):
        """

        # assume that
        :param n_states: Number of states
        :param n_actions: Number of actions
        :param alpha0: Prior weight for uniform Dirichlet
        :param mu0: Prior mean rewards
        :param tau0: Precision of prior mean rewards
        :param tau: Precision of reward noise
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau
        self.executor = VecRandomTabularMDPEnv(n_envs=1, env=self)
        self.reset_trial()

    def reset_trial(self):
        return self.executor.reset_trial([True])[0]

    def reset(self):
        return self.executor.reset([True])[0]

    def step(self, action):
        next_obses, rewards, dones, infos = self.executor.step([action], max_path_length=None)
        return Step(next_obses[0], rewards[0], dones[0], **{k: v[0] for k, v in infos.items()})

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs):
        return VecRandomTabularMDPEnv(n_envs=n_envs, env=self)

    @property
    def observation_space(self):
        return Discrete(self.n_states)

    @property
    def action_space(self):
        return Discrete(self.n_actions)


class VecRandomTabularMDPEnv(object):
    def __init__(self, n_envs, env):
        self.n_envs = n_envs
        self.env = env
        self.states = np.zeros((self.n_envs,), dtype=np.int)
        self.Rs = np.zeros((self.n_envs, self.env.n_states, self.env.n_actions))
        self.Ps = np.zeros((self.n_envs, self.env.n_states, self.env.n_actions, self.env.n_states))
        self.ts = np.zeros((self.n_envs,))
        self.reset_trial([True] * self.n_envs)

    @property
    def num_envs(self):
        return self.n_envs

    def reset_trial(self, dones, seeds=None):
        dones = np.cast['bool'](dones)
        self.states[dones] = 0
        size = (int(np.sum(dones)), self.env.n_states, self.env.n_actions)
        if seeds is None or True:
            self.Rs[dones] = np.ones(size) * self.env.mu0 + np.random.normal(size=size) * 1. / np.sqrt(self.env.tau0)
            self.Ps[dones] = np.random.dirichlet(self.env.alpha0 * np.ones(self.env.n_states, dtype=np.float32),
                                                 size=size)
        else:
            for done_idx, seed in zip(np.where(dones)[0], seeds):
                with using_seed(seed):
                    single_size = (self.env.n_states, self.env.n_actions)
                    self.Rs[done_idx] = np.ones(single_size) * self.env.mu0 + np.random.normal(size=single_size) * 1. \
                                                                              / np.sqrt(self.env.tau0)
                    self.Ps[done_idx] = np.random.dirichlet(
                        self.env.alpha0 * np.ones(self.env.n_states, dtype=np.float32),
                        size=single_size)

        return self.reset(dones)

    def reset(self, dones):
        dones = np.cast['bool'](dones)
        self.states[dones] = 0
        self.ts[dones] = 0
        return self.states

    def step(self, actions, max_path_length):
        ps = self.Ps[np.arange(self.n_envs), self.states, actions]
        next_states = special.weighted_sample_n(ps, np.arange(self.env.n_states))
        reward_means = self.Rs[np.arange(self.n_envs), self.states, actions]
        rewards = reward_means + np.random.normal(size=(self.n_envs,)) * 1. / np.sqrt(self.env.tau)
        self.ts += 1
        self.states = next_states
        if max_path_length is not None:
            dones = self.ts >= max_path_length
        else:
            dones = np.asarray([False] * self.n_envs)
        # if np.any(dones):
        #     self.reset(dones)
        return self.states, rewards, dones, dict()


if __name__ == "__main__":
    env = RandomTabularMDPEnv(n_states=10, n_actions=2)
    vec_env = env.vec_env_executor(n_envs=5)
    vec_env.reset([True] * 5)
    vec_env.step([0] * 5, max_path_length=None)
    vec_env.step([0] * 5, max_path_length=None)
    vec_env.step([0] * 5, max_path_length=None)
