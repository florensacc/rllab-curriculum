from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
from rllab.misc import special
from rllab.algos import util
import numpy as np


class BonusTRPO(TRPO):
    def __init__(self, bonus_evaluator, bonus_coeff, clip_reward=True,*args, **kwargs):
        self.bonus_evaluator = bonus_evaluator
        self.bonus_coeff = bonus_coeff
        self.clip_reward= clip_reward
        super(BonusTRPO, self).__init__(*args, **kwargs)

    def log_diagnostics(self, paths):
        super(BonusTRPO, self).log_diagnostics(paths)
        self.bonus_evaluator.log_diagnostics(paths)

    def process_samples(self, itr, paths):
        logger.log("fitting bonus evaluator before processing...")
        self.bonus_evaluator.fit_before_process_samples(paths)
        logger.log("fitted")

        # recompute the advantages
        # self.bonus_evaluator.
        baselines = []
        returns = []
        for path in paths:
            bonuses = self.bonus_evaluator.predict(path)
            path["bonus_rewards"] = self.bonus_coeff * bonuses
            path["raw_rewards"] = path["rewards"]
            if self.clip_reward:
                path["rewards"] = np.clip(path["raw_rewards"],-1,1) + path["bonus_rewards"]
            else:
                path["rewards"] = path["raw_rewards"] + path["bonus_rewards"]
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            path["raw_returns"] = special.discount_cumsum(path["raw_rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["raw_returns"][0] for path in paths])

            undiscounted_returns = [sum(path["raw_rewards"]) for path in paths]
            undiscounted_bonus_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            raise NotImplementedError

        logger.log("fitting baseline...")
        self.baseline.fit(paths)
        logger.log("fitted")

        logger.log("fitting bonus evaluator after processing...")
        self.bonus_evaluator.fit_after_process_samples(samples_data)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageBonusReturn', np.mean(undiscounted_bonus_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular_misc_stat('Return',undiscounted_returns)

        all_bonus_rewards = np.concatenate([path["bonus_rewards"] for path in paths])
        logger.record_tabular_misc_stat('BonusReward',all_bonus_rewards)

        path_lens = [len(path["rewards"]) for path in paths]
        logger.record_tabular_misc_stat("PathLen",path_lens)

        return samples_data
