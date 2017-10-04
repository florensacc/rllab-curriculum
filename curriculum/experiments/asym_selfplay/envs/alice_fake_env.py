import numpy as np

from rllab import spaces
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger
from rllab.sampler.utils import rollout
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides
from rllab.envs.base import Step
from curriculum.envs.base import FixedStateGenerator


class AliceFakeEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env,
            #env_bob,
            #policy_bob,
            max_path_length,
            alice_bonus,
            alice_factor,
            gamma=0.1,
            stop_threshold=0.9,
            ring_spacing=1,
            init_iter=2
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        #self.env_bob = env_bob
        #self.policy_bob = policy_bob
        self.max_path_length = max_path_length
        self.time = 0
        self.iter = init_iter
        self.ring_spacing = ring_spacing
        self.bob_speed = 0.1 #0.02

        self.alice_bonus = alice_bonus
        self.alice_factor = alice_factor
        self.gamma = gamma
        self.stop_threshold = stop_threshold


    def reset(self, **kwargs):
        ret = self._wrapped_env.reset(**kwargs)
        self.time = 0
        return ret

    def set_iter(self, iter):
        self.iter = iter

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

        # Rather than running bob, analytically compute the time for bob to get to the goal based on the iter
        max_bob_distance = self.iter * self.ring_spacing
        # print("Max bob distance: " + str(max_bob_distance) + ", iter: " + str(self.iter) + ", ring spacing: " + str(self.ring_spacing))
        if np.linalg.norm(bob_start_state) > max_bob_distance:
            # This is farther than what bob knows how to do; return the max time
            # print("Too far for bob")
            t_bob = self.max_path_length - self.time
        else:
            # This is within Bob's range of knowledge; the time for bob is proportional to the distance
            # print("Within bob's range: " + str(self.max_path_length - self.time) + ", " + str(np.linalg.norm(bob_start_state) / self.bob_speed))
            t_bob = min(self.max_path_length - self.time, np.linalg.norm(bob_start_state) / self.bob_speed)
            #t_bob = np.linalg.norm(bob_start_state) / self.bob_speed

        # self.env_bob.update_start_generator(FixedStateGenerator(bob_start_state))
        # path_bob = rollout(self.env_bob, self.policy_bob, max_path_length=max(5, self.max_path_length - self.time), #self.max_path_length, #
        #                    animated=False)
        t_alice = self.time
        #t_bob = path_bob['rewards'].shape[0]
        reward = self.gamma * max(0, self.alice_bonus + t_bob - self.alice_factor * t_alice)
        # print("t_bob: " + str(t_bob) + ", np.linalg.norm(bob_start_state): " + str(np.linalg.norm(bob_start_state)))
        # print("t_alice: " + str(t_alice), " speed: " + str(np.linalg.norm(bob_start_state) / t_alice))
        # print("reward: " + str(reward))
        return reward

    @overrides
    def step(self, action):
        self.time += 1
        wrapped_step = self._wrapped_env.step(action[:-1])
        next_obs, reward, done, info = wrapped_step

        # Determine whether Alice wants to end the episode.
        if np.tanh(action[-1]) > self.stop_threshold:
            done = True
            #logger.log("Alice sampled a stop action at t = " + str(self.time))
        else:
            done = False

        # if self.time >= self.max_path_length and not done:
        #     logger.log("No stop action sampled")

        # Compute the reward for Alice.
        if done or self.time >= self.max_path_length:
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


