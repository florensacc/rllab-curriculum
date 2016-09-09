import tempfile, pickle, subprocess, sys, numpy as np, os
import zerorpc
from .vec_env import VecEnv
from copy import deepcopy

class DummyVecEnv(VecEnv):
    def __init__(self, env, n, k, max_path_length = np.inf):
        self.envs = [deepcopy(env) for _ in range(n*k)]
        self.k = k
        self._action_space = env.action_space
        self._observation_space = env.observation_space        
        self.ts = np.zeros(len(self.envs), dtype='int')        
        self.max_path_length = max_path_length
    def step(self, action_n):
        results = [env.step(a)[:3] for (a,env) in zip(action_n, self.envs)]
        obs, rews, dones = list(map(np.array, list(zip(*results))))
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        for (i, done) in enumerate(dones):
            if done: 
                obs[i] = self.envs[i].reset()
                self.ts[i] = 0        
        return np.array(obs), np.array(rews), np.array(dones)
    def reset(self):        
        results = [env.reset() for env in self.envs]
        return np.array(results)
    @property
    def num_envs(self):
        return len(self.envs)
    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
