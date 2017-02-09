import numpy as np
from rllab.misc import tensor_utils


class VecEnvExecutor(object):
    def __init__(self, envs):
        self.envs = envs
        self._action_space = envs[0].action_space
        self._observation_space = envs[0].observation_space
        self.ts = np.zeros(len(self.envs), dtype='int')

    def step(self, action_n, max_path_length):
        assert len(action_n) == len(self.envs)
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        rewards = np.asarray(rewards)
        self.ts += 1
        if max_path_length is not None:
            dones[self.ts >= max_path_length] = True
        # for (i, done) in enumerate(dones):
        #     if done:
        #         obs[i] = self.envs[i].reset()
        #         self.ts[i] = 0
        return obs, rewards, dones, tensor_utils.stack_tensor_dict_list(env_infos)

    def reset(self, dones, seeds=None, *args, **kwargs):
        assert seeds is None
        dones = np.cast['bool'](dones)
        results = [env.reset() for idx, env in enumerate(self.envs) if dones[idx]]
        self.ts[dones] = 0
        return results

    @property
    def n_envs(self):
        return len(self.envs)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def terminate(self):
        pass
