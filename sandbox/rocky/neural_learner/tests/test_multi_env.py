from io import StringIO

from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.spaces import Discrete, Box
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
import numpy as np

from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor


class SimpleEnv(Env):
    def __init__(self, vectorized):
        self.vectorized = vectorized
        self.executor = VecSimpleEnv(env=self, n_envs=1)

    @property
    def observation_space(self):
        return Box(low=-100, high=100, shape=())

    @property
    def action_space(self):
        return Box(low=-100, high=100, shape=())

    def reset(self):
        return self.executor.reset()[0]

    def reset_trial(self):
        return self.executor.reset_trial()[0]

    def step(self, action):
        next_obses, rewards, dones, infos = self.executor.step([action], max_path_length=None)
        return Step(next_obses[0], rewards[0], dones[0], **{k: v[0] for k, v in infos.items()})

    def vec_env_executor(self, n_envs):
        return VecSimpleEnv(env=self, n_envs=n_envs)


class VecSimpleEnv(object):
    def __init__(self, env, n_envs):
        self.env = env
        self.n_envs = n_envs
        self.states = np.zeros((self.n_envs,))
        self.ts = np.zeros((self.n_envs,))

    def reset(self, dones=None):
        if dones is None:
            dones = np.asarray([True] * self.n_envs)
        else:
            dones = np.cast['bool'](dones)
        self.states[dones] = 0
        self.ts[dones] = 0
        return self.states[dones]

    def reset_trial(self, dones=None):
        return self.reset()

    def step(self, actions, max_path_length):
        self.ts += 1
        self.states += actions
        dones = self.states >= 10
        rewards = np.cast['int'](dones)
        if max_path_length is not None:
            dones[self.ts >= max_path_length] = True
        if np.any(dones):
            self.reset(dones)
        return self.states, rewards, dones, dict()


def test_multi_env():
    multi_env = MultiEnv(wrapped_env=SimpleEnv(vectorized=True), n_episodes=5, episode_horizon=5, discount=0.99)
    multi_env1 = MultiEnv(wrapped_env=SimpleEnv(vectorized=False), n_episodes=5, episode_horizon=5, discount=0.99)

    vec_multi_env = multi_env.vec_env_executor(n_envs=5)
    vec_multi_env1 = VecEnvExecutor(envs=[Serializable.clone(multi_env1) for _ in range(5)])

    assert multi_env.reset() == (0, 0, 0, 1)
    assert multi_env1.reset() == (0, 0, 0, 1)

    sbs = []
    for vec_env in [vec_multi_env, vec_multi_env1]:
        sb = StringIO()
        for obs in vec_env.reset():
            assert tuple(obs) == (0, 0, 0, 1)

        for idx in range(3):
            next_obs, rewards, dones, infos = vec_env.step(np.arange(5), max_path_length=None if idx < 2 else 2)
            sb.write("next_obs=")
            sb.write(str([int(x) for x in np.asarray(next_obs).flat]))
            sb.write("\n")
            sb.write("rewards=")
            sb.write(str([int(x) for x in rewards.flat]))
            sb.write("\n")
            sb.write("dones=")
            sb.write(str([int(x) for x in dones.flat]))
            sb.write("\n")

        sbs.append(sb.getvalue())

    assert sbs[0] == sbs[1]


if __name__ == "__main__":
    test_multi_env()
