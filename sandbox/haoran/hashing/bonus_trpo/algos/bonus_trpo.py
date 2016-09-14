from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger
from rllab.misc import special
from rllab.algos import util
import numpy as np


class BonusTRPO(TRPO):
    def __init__(self, bonus_evaluator, bonus_coeff, extra_bonus_evaluator, clip_reward=True,*args, **kwargs):
        self.bonus_evaluator = bonus_evaluator
        self.extra_bonus_evaluator = extra_bonus_evaluator
        self.bonus_coeff = bonus_coeff
        self.clip_reward= clip_reward
        super(BonusTRPO, self).__init__(*args, **kwargs)

    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
            bonus_evaluator=self.bonus_evaluator,
        )


    def log_diagnostics(self, paths):
        super(BonusTRPO, self).log_diagnostics(paths)
        self.bonus_evaluator.log_diagnostics(paths)

    def process_samples(self, itr, paths):
        logger.log("fitting bonus evaluator before processing...")
        self.bonus_evaluator.fit_before_process_samples(paths)
        self.extra_bonus_evaluator.fit_before_process_samples(paths)
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
        if self.env.wrapped_env.resetter is not None:
            self.env.wrapped_env.resetter.update(paths)

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
            if "prior_reward" in paths[0]["env_infos"]:
                undiscounted_returns = [
                    R + sum(path["env_infos"]["prior_reward"])
                    for R,path in zip(undiscounted_returns, paths)
                ]

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
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["raw_rewards"]) for path in paths]
            if "prior_reward" in paths[0]["env_infos"]:
                undiscounted_returns = [
                    R + sum(path["env_infos"]["prior_reward"])
                    for R,path in zip(undiscounted_returns, paths)
                ]
            undiscounted_bonus_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

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

        # Log info for trajs whose initial states are not modified by the resetter
        if self.env.wrapped_env.resetter is not None:
            test_paths = [
                path for path in paths
                if path["env_infos"]["use_default_reset"][0] == True
            ]
            if len(test_paths) > 0 and type(self.env.wrapped_env.resetter).__name__ != "AtariSaveLoadResetter":
                test_average_discounted_return = \
                    np.mean([path["returns"][0] for path in test_paths])

                test_undiscounted_returns = [sum(path["raw_rewards"]) for path in test_paths]
                test_undiscounted_bonus_returns = [sum(path["rewards"]) for path in test_paths]

                logger.record_tabular('TestAverageDiscountedReturn',
                          test_average_discounted_return)
                logger.record_tabular('TestAverageBonusReturn', np.mean(test_undiscounted_bonus_returns))
                logger.record_tabular('TestNumTrajs', len(test_paths))
                logger.record_tabular_misc_stat('TestReturn',test_undiscounted_returns)

                test_all_bonus_rewards = np.concatenate([path["bonus_rewards"] for path in test_paths])
                logger.record_tabular_misc_stat('TestBonusReward',test_all_bonus_rewards)

                test_path_lens = [len(path["rewards"]) for path in test_paths]
                logger.record_tabular_misc_stat("TestPathLen",test_path_lens)
        return samples_data
