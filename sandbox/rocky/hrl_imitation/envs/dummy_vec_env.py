

import numpy as np
import pickle as pickle


class DummyVecEnv(object):
    def __init__(self, env, n, max_path_length=np.inf, envs=None):
        if envs is None:
            envs = [pickle.loads(pickle.dumps(env)) for _ in range(n)]
        self.envs = envs
        self._action_space = env.action_space
        self._observation_space = env.observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')
        self.max_path_length = max_path_length

    def step(self, action_n):
        results = [env.step(a)[:3] for (a, env) in zip(action_n, self.envs)]
        obs, rews, dones = list(map(np.asarray, list(zip(*results))))
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done:
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0
        return np.asarray(obs), np.asarray(rews), np.asarray(dones), dict()

    def reset(self):
        results = [env.reset() for env in self.envs]
        return np.asarray(results)

    @property
    def num_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space
