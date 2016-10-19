from rllab.envs.base import Env, Step
from rllab.misc import logger
from rllab.misc.ext import using_seed
from rllab.spaces.discrete import Discrete
import numpy as np

from sandbox.rocky.analogy.utils import unwrap


class CopyEnv(Env):
    def __init__(self, min_seq_length=5, max_seq_length=5, n_elems=2, seed=None, target_seed=None):
        """
        :param min_seq_length: Minimum length of sequence (inclusive)
        :param max_seq_length: Maximum length of sequence (inclusive)
        :param n_elems: Number of unique elements appearing in the sequence.
        :param seed: seed for randomized initial conditions (not used in this environment)
        :param target_seed: seed for randomized tasks (in this case, the sequence to be copied)
        :return:
        """
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.n_elems = n_elems
        self.seed = seed
        self.target_seed = target_seed
        self.targets = None
        self.t = 0
        self.reset()

    def reset_trial(self):
        seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed = seed
        target_seed = np.random.randint(np.iinfo(np.int32).max)
        self.target_seed = target_seed
        return self.reset()

    def reset(self):
        # generate binary vectors of this long
        with using_seed(self.target_seed):
            seq_length = np.random.randint(low=self.min_seq_length, high=self.max_seq_length + 1)
            targets = np.random.randint(low=0, high=self.n_elems, size=(seq_length,))
            self.targets = targets
            self.t = 0
        return self.get_current_obs()

    def step(self, action):
        if action == self.targets[self.t]:
            self.t += 1
            return Step(self.get_current_obs(), reward=1, done=self.t >= len(self.targets), **self.get_env_info())
        else:
            return Step(self.get_current_obs(), reward=-100, done=True, **self.get_env_info())

    def get_current_obs(self):
        return 0

    @property
    def observation_space(self):
        return Discrete(n=1)

    @property
    def action_space(self):
        return Discrete(n=self.n_elems)

    def log_analogy_diagnostics(self, eval_paths, log_envs):
        returns = np.asarray([np.sum(p["rewards"]) for p in eval_paths])
        seq_lens = np.asarray([p["env_infos"]["n_targets"][0] for p in eval_paths])
        success = np.mean(seq_lens == returns)
        logger.record_tabular('SuccessRate', success)

    def get_env_info(self):
        return dict(n_targets=len(self.targets))


class CopyPolicy(object):
    def __init__(self, env):
        self.env = unwrap(env)

    def get_action(self, _obs):
        return self.env.targets[self.env.t], dict()

    def reset(self):
        pass
