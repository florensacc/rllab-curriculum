


import numpy as np

from sandbox.rocky.tf.algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from rllab.misc import logger

from rllab.misc import special
from rllab.algos import util


class PredictionBonusEvaluator(object):

    def __init__(self, env_spec, regressor_cls=None):
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        if regressor_cls is None:
            regressor_cls = BernoulliMLPRegressor
        self.regressor = regressor_cls(
            input_shape=(obs_dim+action_dim,),
            output_dim=obs_dim,
            name="state_regressor"
        )
        self.bonus_scale = 1.

    def predict(self, path):
        xs = np.concatenate([path["observations"], path["actions"]], axis=1)[:-1]
        ys = path["observations"][1:]
        return np.append(-self.regressor.predict_log_likelihood(xs, ys) / (self.bonus_scale + 1e-8), 0)

    def fit(self, samples_data):
        paths = samples_data["paths"]
        observations = np.concatenate([p["observations"][:-1] for p in paths])
        actions = np.concatenate([p["actions"][:-1] for p in paths])
        next_observations = np.concatenate([p["observations"][1:] for p in paths])
        xs = np.concatenate([observations, actions], axis=1)
        self.regressor.fit(xs, next_observations)
        all_bonuses = -self.regressor.predict_log_likelihood(xs, next_observations)
        self.bonus_scale = np.median(all_bonuses)

    def log_diagnostics(self, samples_data):
        bonuses = np.concatenate(list(map(self.predict, samples_data["paths"])))
        logger.record_tabular("AverageBonus", np.mean(bonuses))
        logger.record_tabular("MaxBonus", np.max(bonuses))
        logger.record_tabular("MinBonus", np.min(bonuses))
        logger.record_tabular("StdBonus", np.std(bonuses))


class BonusTRPO(TRPO):
    def __init__(self, bonus_evaluator, bonus_coeff, *args, **kwargs):
        self.bonus_evaluator = bonus_evaluator
        self.bonus_coeff = bonus_coeff
        super(BonusTRPO, self).__init__(*args, **kwargs)

    def log_diagnostics(self, samples_data):
        super(BonusTRPO, self).log_diagnostics(samples_data)
        self.bonus_evaluator.log_diagnostics(samples_data)

    def process_samples(self, itr, paths):
        # recompute the advantages
        # self.bonus_evaluator.
        baselines = []
        returns = []
        for path in paths:
            bonuses = self.bonus_evaluator.predict(path)
            path["raw_rewards"] = path["rewards"]
            path["rewards"] = path["raw_rewards"] + self.bonus_coeff * bonuses
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

        logger.log("fitting bonus evaluator...")
        self.bonus_evaluator.fit(samples_data)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('AverageBonusReturn', np.mean(undiscounted_bonus_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
