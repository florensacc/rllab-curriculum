import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.sampler.utils import rollout
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from sandbox.young_clgan.envs.base import FixedStateGenerator


class AliceEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            env_bob,
            policy_bob,
            max_path_length,
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        self.env_bob = env_bob
        self.policy_bob = policy_bob
        self.max_path_length = max_path_length
        self.time = 0

        self.alice_bonus = 10
        self.alice_factor = 0.5
        self.gamma = 0.1


    def reset(self, **kwargs):
        ret = self._wrapped_env.reset(**kwargs)
        self.time = 0
        return ret

    @property
    @overrides
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            wrapped_low = np.append(self._wrapped_env.action_space.low,[-1])
            wrapped_high =  np.append(self._wrapped_env.action_space.high, [1])
            return spaces.Box(wrapped_low, wrapped_high)
        else:
            raise NotImplementedError

    def compute_alice_reward(self, next_obs):
        alice_end_obs = next_obs
        bob_start_state = self._obs2start_transform(alice_end_obs)
        self.env_bob.update_start_generator(FixedStateGenerator(bob_start_state))
        path_bob = rollout(self.env_bob, self.policy_bob, max_path_length=max(1, self.max_path_length - self.time), #self.max_path_length,
                           animated=False)
        t_alice = self.time
        t_bob = path_bob['rewards'].shape[0]
        reward = self.gamma * max(0, self.alice_bonus + t_bob - self.alice_factor * t_alice)
        return reward

    @overrides
    def step(self, action):
        self.time += 1
        wrapped_step = self._wrapped_env.step(action[:-1])
        next_obs, reward, done, info = wrapped_step

        # Determine whether Alice wants to end the episode.
        if np.tanh(action[-1])>0.9:
            done = True
        else:
            done = False

        # Compute the reward for Alice.
        if done or self.time > self.max_path_length:
            # Alice is done here; we need to run Bob!
            reward = self.compute_alice_reward(next_obs)
        else:
            reward = 0

        return Step(next_obs, reward, done, **info)

    def get_current_obs(self):
        return self._wrapped_env.get_current_obs()

    def __str__(self):
        return "Wrapped with stop action: %s" % self._wrapped_env

    @overrides
    def log_diagnostics(self, paths, n_traj=1, *args, **kwargs):
        pass


