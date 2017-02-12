import copy

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.misc import logger
from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv, CachedWorldBuilder
import random
import gpr.env
import numpy as np
from sandbox.rocky.tf.envs.vec_env_executor import VecEnvExecutor


class TrajResetEnv(Env, Serializable):
    """
    Given a gpr env and some trajectories, construct a new environment where
    the initial state distribution is any state along the trajectory, and the
    corresponding terminal state is a few time steps ahead along the trajectory
    """

    def __init__(self, env: GprEnv, paths, k, reward_type='state'):
        Serializable.quick_init(self, locals())
        self.env = env
        self.paths = paths
        self.k = k
        self.init_state = None
        self.final_state = None
        self.init_obs = None
        self.final_obs = None
        self.reward_type = reward_type

    def reset(self):
        path = random.choice(self.paths)
        # need access to the underlying state
        xs = path["env_infos"]["x"]
        obs = path["observations"]
        xid = np.random.randint(low=0, high=max(1, len(xs) - self.k))
        self.init_state = xs[xid]
        self.init_obs = obs[xid]
        self.final_state = xs[min(xid + self.k, len(xs) - 1)]
        self.final_obs = obs[min(xid + self.k, len(xs) - 1)]
        return copy.deepcopy(self.env.gpr_env.reset_to(self.init_state))

    def step(self, action):
        next_obs, reward, done, infos = self.env.step(action)
        if self.reward_type == 'state':
            reward = -np.linalg.norm(self.env.gpr_env.x - self.final_state)
        elif self.reward_type == 'obs':
            flat_next_obs = self.observation_space.flatten(next_obs)
            reward = -np.linalg.norm(flat_next_obs - self.final_obs)
        else:
            raise NotImplementedError
        return next_obs, reward, done, infos

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def vectorized(self):
        return True

    def log_diagnostics(self, paths):
        final_rewards = [p["rewards"][-1] for p in paths]
        logger.record_tabular_misc_stat('FinalReward', final_rewards, placement='front')

    def vec_env_executor(self, n_envs):
        gpr_env = self.env.gpr_env
        envs = [
            TrajResetEnv(
                env=GprEnv(
                    env_name=self.env.env_name,
                    gpr_env=gpr.env.Env(world_builder=CachedWorldBuilder(gpr_env), reward=gpr_env.reward,
                                        horizon=gpr_env.horizon, task_id=gpr_env.task_id,
                                        delta_reward=gpr_env.delta_reward,
                                        delta_obs=gpr_env.delta_obs)
                ),
                paths=self.paths,
                k=self.k
            )
            for _ in range(n_envs)
            ]
        return VecEnvExecutor(envs=envs)

    def render(self, *args, **kwargs):
        self.env.render(*args, **kwargs)
