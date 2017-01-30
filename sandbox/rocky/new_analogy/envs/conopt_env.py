import random

from cached_property import cached_property
from gym.utils import seeding

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.envs.gym_env import convert_gym_space
import conopt.env
import conopt.envs
from rllab.envs.base import Step
import numpy as np
from rllab.misc import logger
from rllab.misc.ext import using_seed


class ConoptEnv(Env, Serializable):
    def __init__(self, env_name, xinits=None, seed=None, task_id=None):
        Serializable.quick_init(self, locals())
        self.env_name = env_name
        with using_seed(seed):
        # if seed is None:
        #     seed = np.random.randint(0, np.iinfo(np.int32).max)
        #     rng = np.random.RandomState()
        #     rng.seed(seed)
        # else:
        #     rng = np.random
        # # seeding.np_random
        # print(seed)
            expr = conopt.envs.load(env_name)
            if task_id is not None:
                self.conopt_env = expr.make(task_id)
            elif hasattr(expr, "task_id_iterator"):
                task_id = np.random.choice(list(expr.task_id_iterator()))
                self.conopt_env = expr.make(task_id)
            else:
                self.conopt_env = expr.make()
            self.xinits = xinits
            self.reset()

    @cached_property
    def observation_space(self):
        return convert_gym_space(self.conopt_env.observation_space)

    @cached_property
    def action_space(self):
        return convert_gym_space(self.conopt_env.action_space)

    def reset(self):
        if self.xinits is not None:
            xinit = random.choice(self.xinits)
            xinit = xinit[:self.conopt_env.world.dimx]
            return self.conopt_env.reset_to(xinit)
        else:
            return self.conopt_env.reset()

    def step(self, action):
        try:
            action = np.asarray(action, dtype=np.float)
            obs, rew, done, info = self.conopt_env.step(action)
            info = dict()
            assert np.max(np.abs(obs)) < 1000
        except AssertionError:
            # wtf...
            obs = self.observation_space.default_value
            rew = 0  # -1000
            done = True
            info = dict()
        if hasattr(self.conopt_env, "task_id_iterator"):
            info["task_id"] = self.conopt_env.task_id
        return Step(observation=obs, reward=rew, done=done, **info)

    def render(self, mode='human', close=False):
        return self.conopt_env.render(mode=mode, close=close)

    def log_diagnostics(self, paths):
        if self.env_name in ["I1_copter_3_targets", "TF2"]:
            success_threshold = 4
        else:
            raise NotImplementedError
        # import ipdb; ipdb.set_trace()
        if "raw_rewards" in paths[0]:
            logger.record_tabular('SuccessRate', np.mean([p["raw_rewards"][-1] >= success_threshold for p in paths]))
        else:
            logger.record_tabular('SuccessRate', np.mean([p["rewards"][-1] >= success_threshold for p in paths]))
