import random

from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.core.serializable import Serializable
from rllab.misc import logger as root_logger
import numpy as np
from rllab.misc import logger

from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv, VecMultiEnv

BIG = 50000


class MultiResetterEnv(MultiEnv, Serializable):
    def __init__(
            self,
            wrapped_env,
            n_episodes,
            episode_horizon,
            discount,
            # The portion of states that are reset according to the previous iteration
            reset_ratio=0.5,
            recompute_hidden_states=False,
    ):
        Serializable.quick_init(self, locals())
        self.reset_ratio = reset_ratio
        self.state_pool = None
        self.recompute_hidden_states = recompute_hidden_states
        self.policy = None
        MultiEnv.__init__(
            self,
            wrapped_env=wrapped_env,
            n_episodes=n_episodes,
            episode_horizon=episode_horizon,
            discount=discount
        )

    def vec_env_executor(self, n_envs):
        return VecMultiResetterEnv(n_envs=n_envs, env=self)

    def log_diagnostics(self, paths, *args, **kwargs):
        state_pool = []
        raw_obs_space, action_space, _, _ = self.observation_space.components
        raw_obs_dim = raw_obs_space.flat_dim
        action_dim = action_space.flat_dim

        original_paths = [p for p in paths if not p["env_infos"]["pool_reset"][0]]

        for p in original_paths:
            done_ids = np.where(p["observations"][:, -1])[0]

            if self.recompute_hidden_states:
                assert self.policy is not None
                prev_states = p["agent_infos"]["prev_state"]
                hiddens = self.policy.f_hiddens([p["observations"]], [prev_states[0]])[0]
                new_prev_states = np.concatenate([prev_states[:1], hiddens[:-1]], axis=0)
                done_prev_states = new_prev_states[done_ids]
            else:
                done_prev_states = p["agent_infos"]["prev_state"][done_ids]

            done_episode_ids = p["env_infos"]["episode_id"][done_ids]
            done_trial_seeds = p["env_infos"]["trial_seed"][done_ids]

            done_last_actions = p["observations"][done_ids, raw_obs_dim:raw_obs_dim + action_dim]
            done_last_rewards = p["observations"][done_ids, raw_obs_dim + action_dim]
            done_last_terminals = p["observations"][done_ids, -1]

            assert np.all(done_last_terminals)

            state_pool.extend(zip(done_prev_states, done_episode_ids, done_trial_seeds, done_last_actions,
                                  done_last_rewards, done_last_terminals))

        self.state_pool = np.asarray(state_pool)

        returns = [np.sum(p["rewards"]) for p in original_paths]

        logger.record_tabular_misc_stat(key="OriginalReturn", values=returns, placement="front")


class VecMultiResetterEnv(VecMultiEnv):
    def __init__(self, n_envs, env):
        VecMultiEnv.__init__(self, n_envs=n_envs, env=env)
        self.is_pool_reset = np.zeros((self.n_envs,), dtype=np.int)
        self.last_pool_prev_states = None
        self.last_pool_reset_mask = None

    def reset(self, dones=None):
        if self.env.state_pool is None or len(self.env.state_pool) == 0:
            return VecMultiEnv.reset(self, dones)
        else:
            if dones is None:
                dones = np.asarray([True] * self.n_envs)
            else:
                dones = np.cast['bool'](dones)
            reset_mask = np.random.uniform(size=self.n_envs) < self.env.reset_ratio
            original_reset = np.logical_and(dones, 1 - reset_mask)
            pool_reset = np.logical_and(dones, reset_mask)
            self.is_pool_reset[original_reset] = 0
            self.is_pool_reset[pool_reset] = 1
            if np.any(original_reset):
                original_reset_response = VecMultiEnv.reset(self, original_reset)
            else:
                original_reset_response = []
            if np.any(pool_reset):
                # for these in the reset mask, we will sample from the state pool; the rest

                pool_samples = self.env.state_pool[
                    np.random.choice(np.arange(len(self.env.state_pool)), size=np.sum(pool_reset))
                ]
                prev_states, episode_ids, trial_seeds, last_actions, last_rewards, last_terminals = \
                    [np.asarray(x) for x in zip(*pool_samples)]
                self.last_actions[pool_reset] = last_actions
                self.last_rewards[pool_reset] = last_rewards
                self.last_terminals[pool_reset] = last_terminals
                self.cnt_episodes[pool_reset] = episode_ids
                self.episode_t[pool_reset] = 0
                self.ts[pool_reset] = 0
                self.trial_seeds[pool_reset] = trial_seeds
                self.last_pool_prev_states = prev_states
                self.last_pool_reset_mask = pool_reset
                pool_reset_response = self.convert_obs(self.vec_env.reset_trial(pool_reset, seeds=trial_seeds))
            else:
                pool_reset_response = []
                self.last_pool_reset_mask = np.zeros((self.n_envs,), dtype=np.bool)
                self.last_pool_prev_states = []
            original_reset_response_with_ids = list(zip(np.where(original_reset)[0], original_reset_response))
            pool_reset_response_with_ids = list(zip(np.where(pool_reset)[0], pool_reset_response))

            joint_response_with_ids = sorted(original_reset_response_with_ids + pool_reset_response_with_ids,
                                             key=lambda x: x[0])
            joint_response = [x[1] for x in joint_response_with_ids]

            return joint_response

    def step(self, actions, max_path_length):
        obs, rewards, dones, infos = VecMultiEnv.step(self, actions, max_path_length)
        return obs, rewards, dones, dict(infos, pool_reset=np.copy(self.is_pool_reset))

    def handle_policy_reset(self, policy, dones):
        if np.any(dones):
            if self.env.state_pool is None or not np.any(self.last_pool_reset_mask):
                policy.reset(dones)
            else:
                # check that last_pool_reset_mask should be a subset of dones
                assert hasattr(policy, "stateful_reset")
                assert not np.any(np.logical_and(self.last_pool_reset_mask, np.logical_not(dones)))
                original_reset = np.logical_and(np.logical_not(self.last_pool_reset_mask), dones)
                if np.any(original_reset):
                    policy.reset(original_reset)
                policy.stateful_reset(self.last_pool_reset_mask, self.last_pool_prev_states)
                self.last_pool_reset_mask = np.zeros((self.n_envs,), dtype=np.bool)
                self.last_pool_prev_states = []
