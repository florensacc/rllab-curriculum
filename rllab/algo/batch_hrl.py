from rllab.algo.batch_polopt import BatchPolopt
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from rllab.misc import logger
import numpy as np


class BatchHRL(BatchPolopt, Serializable):

    def __init__(self, high_algo, low_algo, **kwargs):
        super(BatchHRL, self).__init__(**kwargs)
        Serializable.quick_init(self, locals())
        self._high_algo = high_algo
        self._low_algo = low_algo

    @property
    def high_algo(self):
        return self._high_algo

    @property
    def low_algo(self):
        return self._low_algo

    @overrides
    def init_opt(self, mdp, policy, baseline, **kwargs):
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            high_opt_info = \
                self.high_algo.init_opt(mdp=mdp.high_mdp, policy=policy.high_policy, baseline=baseline.high_baseline)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            low_opt_info = \
                self.low_algo.init_opt(mdp=mdp.low_mdp, policy=policy.low_policy, baseline=baseline.low_baseline)
        return dict(
            high=high_opt_info,
            low=low_opt_info
        )

    def optimize_policy(self, itr, policy, samples_data, opt_info, **kwargs):
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
                opt_info["high"] = self.high_algo.optimize_policy(
                    itr, policy.high_policy, samples_data["high"], opt_info["high"])
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
                opt_info["low"] = self.low_algo.optimize_policy(
                    itr, policy.low_policy, samples_data["low"], opt_info["low"])
        return opt_info

    @overrides
    def get_itr_snapshot(self, itr, mdp, policy, baseline, samples_data, opt_info, **kwargs):
        return dict(
            itr=itr,
            policy=policy,
            baseline=baseline,
            mdp=mdp,
        )

    def process_samples(self, itr, paths, mdp, policy, baseline):
        """
        Transform paths to high_paths and low_paths, and dispatch them to the high-level and low-level algorithms
        respectively.
        """
        high_paths = []
        low_paths = []
        for path in paths:
            pdists = path['pdists']
            observations = path['observations']
            rewards = path['rewards']
            actions = path['actions']
            flat_observations = np.reshape(observations, (observations.shape[0], -1))
            high_pdists, low_pdists, subgoals = policy.split_pdists(pdists)
            high_path = dict(
                rewards=rewards,
                pdists=high_pdists,
                actions=subgoals,
                observations=observations,
            )
            low_path = dict(
                rewards=rewards,
                pdists=low_pdists,
                actions=actions,
                observations=np.concatenate([flat_observations, subgoals], axis=1)
            )
            high_paths.append(high_path)
            low_paths.append(low_path)
        with logger.tabular_prefix('Hi_'), logger.prefix('Hi | '):
            high_samples_data = self.high_algo.process_samples(
                itr, high_paths, mdp.high_mdp, policy.high_policy, baseline.high_baseline)
        with logger.tabular_prefix('Lo_'), logger.prefix('Lo | '):
            low_samples_data = self.low_algo.process_samples(
                itr, low_paths, mdp.low_mdp, policy.low_policy, baseline.low_baseline)
        return dict(
            high=high_samples_data,
            low=low_samples_data
        )
