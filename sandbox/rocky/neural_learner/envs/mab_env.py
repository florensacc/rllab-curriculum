from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.discrete import Discrete
import numpy as np


class MABEnv(Env, Serializable):
    # start with a bernoulli
    def __init__(self, n_arms=10, arm_dist="bernoulli"):
        Serializable.quick_init(self, locals())
        assert arm_dist == "bernoulli"
        self.arm_dist = arm_dist
        self.n_arms = n_arms
        self.arm_means = None
        self.executor = VecMAB(n_envs=1, env=self)
        self.reset()

    def reset(self):
        return self.executor.reset(dones=[True])[0]

    def reset_trial(self):
        return self.executor.reset_trial(dones=[True])[0]

    @property
    def observation_space(self):
        return Discrete(1)

    @property
    def action_space(self):
        return Discrete(self.n_arms)

    def step(self, action):
        next_obses, rewards, dones, infos = self.executor.step([action], max_path_length=None)
        return Step(next_obses[0], rewards[0], dones[0], **{k: v[0] for k, v in infos.items()})

    @property
    def vectorized(self):
        return True

    def vec_env_executor(self, n_envs):
        return VecMAB(n_envs=n_envs, env=self)


class VecMAB(object):
    def __init__(self, n_envs, env):
        self.n_envs = n_envs
        self.env = env
        self.arm_means = np.zeros((self.n_envs, self.env.n_arms))
        self.ts = np.zeros((self.n_envs,))
        self.reset_trial(np.asarray([True] * self.n_envs))

    def reset_trial(self, dones):
        dones = np.cast['bool'](dones)
        self.ts[dones] = 0
        if self.env.arm_dist == "bernoulli":
            self.arm_means[dones] = np.random.uniform(size=(int(np.sum(dones)), self.env.n_arms))
        else:
            raise NotImplementedError
        return self.reset(dones)

    def reset(self, dones):
        dones = np.cast['bool'](dones)
        return self.get_current_obs()[dones]

    def get_current_obs(self):
        return np.zeros((self.n_envs,), dtype=np.int)

    def step(self, actions, max_path_length):
        selected_arm_means = self.arm_means[np.arange(self.n_envs), actions]
        rewards = np.random.binomial(1, selected_arm_means)
        self.ts += 1
        if max_path_length is not None:
            dones = self.ts >= max_path_length
        else:
            dones = np.asarray([False] * self.n_envs)
        if np.any(dones):
            self.reset(dones)
        if np.shape(rewards) == ():
            rewards = np.asarray([rewards])
        return self.get_current_obs(), rewards, dones, dict()
